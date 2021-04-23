import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel
from typing import Dict, List, Optional, Tuple
import numpy as np

from facenet_pytorch import InceptionResnetV1
import microsoftvision
from networks import pit

#  Model
# There are two main models, the VisionEncoder and the TextEncoder which have resnet and distilbert as backbones. 
# In order to make it multi-lingual, we simply choose the distilbert-multilingual model and that's it! 
# No need to specifically train on non-english words as you will soon see.
# The Projection module, takes the embeddings from vision and text encoders and projects them into 512 dimensional space.
# Two things to note:
# We have frozen both the text and vision encoder backbones and do not retrain their weights at all.
# For both encoders the final output is normalised to be of unit length.
class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float=0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds

class VisionEncoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        if args.vision_model == 'vggface':
            # For a model pretrained on VGGFace2
            base = InceptionResnetV1(pretrained='vggface2').train()
            d_in = 512
        elif args.vision_model == 'ms_vision':
            base = microsoftvision.models.resnet50(pretrained=True)
            d_in = 2048
        elif args.vision_model == 'pit':
            base = pit.pit_s(pretrained=False, image_size=args.img_size, num_classes=0)
            d_in = 576            
        else:
            base = models.resnet50(pretrained=True)
            d_in = base.fc.in_features

        self.base = base
        if args.freeze:
            for p in self.base.parameters():
                p.requires_grad = False

        self.projection = Projection(d_in, args.embed_dim)
        

    def forward(self, x):
        projected_vec = self.projection(self.base(x))
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len

class TextEncoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.base = AutoModel.from_pretrained(args.text_model)
        self.projection = Projection(args.transformer_embed_dim, args.embed_dim)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        # out = self.base(**x)[0]
        input_ids = x['input_ids'].squeeze(dim=1)
        attention_mask = x['attention_mask'].squeeze(dim=1)
        out = self.base(input_ids, attention_mask=attention_mask)[0]

        out = out[:, 0, :]  # get CLS token output
        projected_vec = self.projection(out)
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len


# CLIP loss function
# For someone like me who hasn't played around with contrastive loss, this was the most interesting part.
# We know that we want the vectors of the corresponding image and the text to line up. 
# Which means that the dot product has to be as close to one as possible. For everything else we need to push it towards 0.
# Therfore for a given caption, we take the softmax of the dot products across all images, and then take cross entropy loss. 
# Similarly for a given image, we repeat the process across all captions. We average these two losses.
# In terms of which element is the true positive within a batch, remember that we are sending image, caption pairs already lined up. 
# Therefore we want all the diagonal elements to line up while all off-diagonal elements we want to push towards zero.
def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()
    
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0

def metrics(similarity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(len(similarity)).to(similarity.device)
    img2cap_match_idx = similarity.argmax(dim=1)
    cap2img_match_idx = similarity.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    cap_acc = (cap2img_match_idx == y).float().mean()

    return img_acc, cap_acc


# Model
# If you haven't used pytorch lightning before, the benefit is that you do not need to stress about which device to put it in, 
# remembering to zero the optimizer etc. All of that is taken care of. 
# Just simply specify the training and validation steps, along with the optimizer and you are good to go.
# The other benefit that I really like is logging. 
# You just need to write self.log("name", metric_to_track) and it will log to tensorboard by default, or any other kind of logger for that matter.
class LitModel(pl.LightningModule):
    def __init__(self, 
                 args,
        ) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoder(args)
        self.caption_encoder = TextEncoder(args)
        
        self.lr = args.lr
        self.freeze = args.freeze
        self.best_loss = 100.0


    def common_step(self, batch: Tuple[torch.Tensor, List[str]]) -> torch.Tensor:
        images, text = batch
        # device = images.device
        # text_dev = {k: v.to(self.device) for k, v in text.items()}

        image_embed = self.vision_encoder(images)
        caption_embed = self.caption_encoder(text)
        similarity = caption_embed @ image_embed.T

        loss = clip_loss(similarity)
        img_acc, cap_acc = metrics(similarity)
        return loss, img_acc, cap_acc

    def training_step(
        self, batch: Tuple[torch.Tensor, List[str]], *args: list
    ) -> torch.Tensor:
        loss, img_acc, cap_acc = self.common_step(batch)     
        return {'loss': loss, 'img_acc': img_acc, 'cap_acc': cap_acc}
        
    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_train_img_acc = torch.stack([x["img_acc"] for x in outputs]).mean()
        avg_train_cap_acc = torch.stack([x["cap_acc"] for x in outputs]).mean()
        self.log('train_loss', avg_train_loss)
        self.log('train_img_acc', avg_train_img_acc)
        self.log('train_cap_acc', avg_train_cap_acc)


    def validation_step(
        self, batch: Tuple[torch.Tensor, List[str]], *args: list
    ) -> torch.Tensor:
        loss, img_acc, cap_acc = self.common_step(batch)
        return {'loss': loss, 'img_acc': img_acc, 'cap_acc': cap_acc}

    def validation_epoch_end(self, outputs):
        avg_valid_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_valid_img_acc = torch.stack([x["img_acc"] for x in outputs]).mean()
        avg_valid_cap_acc = torch.stack([x["cap_acc"] for x in outputs]).mean()
        self.log('valid_loss', avg_valid_loss)
        self.log('valid_img_acc', avg_valid_img_acc)
        self.log('valid_cap_acc', avg_valid_cap_acc)

        # save best model
        if avg_valid_loss < self.best_loss:
            self.best_loss = avg_valid_loss
            best_model = f'best_model.epoch-{self.trainer.current_epoch:03d}.best_loss-{avg_valid_loss:.2f}.best_img_acc-{avg_valid_img_acc:.2f}.best_cap_acc{avg_valid_cap_acc:.2f}.ckpt'
            self.trainer.save_checkpoint('./checkpoints_best/'+best_model)


    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.freeze:
            vision_params = {"params": self.vision_encoder.projection.parameters(), "lr": self.lr}
        else:
            vision_params = {"params": self.vision_encoder.parameters(), "lr": self.lr}
        caption_params = {"params": self.caption_encoder.projection.parameters() , "lr": self.lr}
        return torch.optim.Adam([vision_params, caption_params])
