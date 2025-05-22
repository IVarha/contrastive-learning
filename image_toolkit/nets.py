import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torchvision import models
from tqdm.asyncio import tqdm

from .clustering import evaluate_clustering_on_validation_p
from .losses import supervised_nt_xent_loss


class TransformerPatchCluster(nn.Module):
    """
    Transformer-based model for clustering patches from images.
    Args:
        device (str): Device to run the model on device instance itself.
        embed_dim (int): Dimension of the embedding space.
        nhead (int): Number of attention heads in the transformer.
        num_layers (int): Number of transformer encoder layers.
    """
    def __init__(self, embed_dim=128, nhead=4, num_layers=4):
        """Initialize the model."""

        super().__init__()

        # CNN Backbone (ResNet18)
        backbone = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])  # [B*N, 512, H', W']
        self.fc = nn.Linear(512, embed_dim)  # Project to transformer input dim

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*2, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, patches):
        feats = self.encoder(patches)          # [B*N, 512, H', W']
        feats = feats.mean(dim=[2, 3])         # [B*N, 512]
        feats = self.fc(feats)                 # [B*N, embed_dim]

        feats = self.transformer(feats)        # [B, N, embed_dim]

        return F.normalize(feats, dim=1)       # for contrastive learning

    def train_model(self, train_loader, val_loader,
                    optimizer, scheduler, device, epochs=10, temperature=0.5):
        """
        Train the model using supervised contrastive learning.
        Args:
        :param train_loader: dataloader for training data
        :param val_loader: dataloader for validation data
        :param optimizer: optimizer for training
        :param scheduler: scheduler for learning rate adjustment
        :param device: Specify the device to run the model on
        :param epochs: Number of epochs to train
        :param temperature: Temperature parameter for contrastive loss
        :return: Validation losses
        """
        val_losses = [0]
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for batch, classes in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                B, N, C, H, W = batch.shape
                batch = batch.view(B * N, C, H, W).to(device)
                labels = torch.arange(B).repeat_interleave(N).to(device)
                # shuffle labels and batch same
                idx = torch.randperm(batch.shape[0])
                batch = batch[idx]
                labels = labels[idx]

                embeddings = self.forward(batch)

                loss = supervised_nt_xent_loss(embeddings, labels, temperature=temperature)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

            self.eval()
            with torch.no_grad():
                avg_ari,_,_ = evaluate_clustering_on_validation_p(val_loader, self, device=device)
                print(f"Epoch [{epoch + 1}/{epochs}], ARI: {avg_ari:.4f}")

                # Save the model if ARI improves
                if avg_ari > max(val_losses):
                    torch.save(self.state_dict(), f"best_model_epoch_{epoch + 1}.pth")
                    print(f"Model saved at epoch {epoch + 1} with ARI: {avg_ari:.4f}")
                val_losses.append(avg_ari)
                if scheduler is not None:
                    # Step the scheduler with the average ARI
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(avg_ari)
                    else:
                        scheduler.step()
                    print("Current learning rate:", scheduler.get_last_lr())

        return val_losses[1:]  # Exclude the initial value

    def load_weights(self, path,device):
        """
        Load model weights from a file.
        Args:
            path (str): Path to the weights file.
        """
        self.load_state_dict(torch.load(path, map_location=device))
        print(f"Weights loaded from {path}")