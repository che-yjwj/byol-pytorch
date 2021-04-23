import torch
from torchvision import models
import pytorch_lightning as pl

# from networks import vit
# from networks import byol_pytorch
from networks.byol_pytorch import BYOL
from networks.vit import ViT

# pytorch lightning module

class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        if args.network == 'vit':
            backbone = ViT(
                image_size = args.img_size,
                patch_size = 4,
                num_classes = 100,
                dim = 512,
                depth = 6,
                heads = 8,
                mlp_dim = 1024
            )
            hidden_layer = 'to_latent'

        else:
            backbone = models.resnet50(pretrained=True)
            hidden_layer = 'avgpool'


        self.learner  = BYOL(
            backbone,
            image_size = args.img_size,
            hidden_layer = hidden_layer,
            projection_size = 256,
            projection_hidden_size = 2048,
            moving_average_decay = 0.99
        )
        self.lr = args.lr


    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log('train_loss', avg_train_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

