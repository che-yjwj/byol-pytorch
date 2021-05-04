import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import pytorch_lightning as pl
import numpy as np

from networks.byol_pytorch import BYOL
from networks.vit import ViT

from sklearn.linear_model import LogisticRegression


def eval_logistic_regression(embeddings, labels, eval_classifier, limit_ratio):
    indices = np.arange(embeddings.shape[0])
    np.random.shuffle(indices)
    limit_len = int(embeddings.shape[0] * limit_ratio)
    indices = indices[:limit_len]
    embeddings = embeddings[indices]
    labels = labels[indices]

    num_split_linear = embeddings.shape[0] // 2
    eval_classifier.fit(embeddings[:num_split_linear], labels[:num_split_linear])
    train_accuracy = eval_classifier.score(embeddings[:num_split_linear], labels[:num_split_linear])
    val_accuracy = eval_classifier.score(embeddings[num_split_linear:], labels[num_split_linear:])
    
    return train_accuracy, val_accuracy


# pytorch lightning module
class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.lr = args.lr


        if args.network == 'vit':
            self.backbone = ViT(
                image_size = args.img_size,
                patch_size = 4,
                num_classes = 100,
                dim = 512,
                depth = 6,
                heads = 8,
                mlp_dim = 1024
            )
            hidden_layer = 'to_latent'
            num_filters  = 512

        else:
            self.backbone = models.resnet50(pretrained=True)
            hidden_layer = 'avgpool'
            num_filters  = 2048
            self.backbone.fc = nn.Identity()


        self.learner  = BYOL(
            self.backbone,
            image_size = args.img_size,
            hidden_layer = hidden_layer,
            projection_size = 256,
            projection_hidden_size = 2048,
            moving_average_decay = 0.99,
            use_momentum = args.use_momentum
        )

        # this classifier is used to compute representation quality each epoch
        # https://github.com/untitled-ai/self_supervised/blob/master/moco.py
        self.eval_classifier = LogisticRegression(max_iter=100, solver="liblinear")


    def forward(self, images):
        return self.learner(images)

    def training_step(self, batch, batch_idx):
        images, _ = batch
        loss = self.forward(images)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log('train_loss', avg_train_loss)


    def validation_step(self, batch, batch_idx):
        images, labels = batch
        with torch.no_grad():
            emb = self.backbone(images)
        return {"emb": emb, "labels": labels}

    def validation_epoch_end(self, outputs):
        embeddings = torch.cat([x["emb"] for x in outputs]).cpu().detach().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).cpu().detach().numpy()

        train_accuracy, val_accuracy = eval_logistic_regression(embeddings, labels, self.eval_classifier, 0.25)

        print(f"Epoch {self.current_epoch} accuracy: train: {train_accuracy:.3f}%, validation: {val_accuracy:.3f}%")
        self.log('logistic_regression_train_acc', train_accuracy)
        self.log('logistic_regression_val_acc', val_accuracy)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
        # multiple lr

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

