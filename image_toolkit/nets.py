import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from torch_geometric.nn import GATConv
from torchvision import models
from tqdm.asyncio import tqdm
from .losses import supervised_nt_xent_loss

def cluster_and_score(embeddings: torch.Tensor, true_labels: list[int],B) -> float:
    """
    Perform HDBSCAN clustering and compute clustering accuracy via ARI.

    Args:
        embeddings (torch.Tensor): Tensor of shape [N, D], patch embeddings.
        true_labels (List[int]): Ground-truth image indices for each patch.

    Returns:
        ari (float): Adjusted Rand Index between predicted and true labels.
    """
    embeddings_np = embeddings.detach().cpu().numpy()
    clusterer =  KMeans(n_clusters=B, random_state=42)
    pred_labels = clusterer.fit_predict(embeddings_np)
    true_labels = np.array(true_labels)
    # print("true_labels ", true_labels)
    # print("pred_labels ", pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    return ari


def evaluate_clustering_on_validation( dataloader,model,device):
    """
    Evaluate clustering performance on validation set.
    Args:
        :param dataloader (DataLoader): DataLoader for validation set.
        :param model (nn.Module): Trained model for generating embeddings
        :param device:
    """
    ari_list = []
    for batch, labels in dataloader:
        batch = batch.to(device)
        with torch.no_grad():

            B, N, C, H, W = batch.shape

            # reshape to [B*N, C, H, W]

            true_labels = labels.numpy().repeat(N)
            #print("true_labels ", true_labels)
            batch = batch.view(B * N, C, H, W)

            # Get embeddings
            embeddings = model(batch)

            #embeddings = embeddings.view(-1, embeddings.shape[-1])
            #print(embeddings.shape)
            # Compute ARI
            ari = cluster_and_score(embeddings, true_labels,B)
            ari_list.append(ari)
            #print(f"Adjusted Rand Index (ARI): {ari:.4f}")

    # Average ARI over all batches
    avg_ari = np.mean(ari_list)
    return avg_ari

def create_patch_graph(embeddings, top_k=5):
    """
    embeddings: Tensor [num_patches, embed_dim]
    returns: edge_index (2, num_edges)
    """
    embeddings = F.normalize(embeddings, dim=1)
    sim_matrix = embeddings @ embeddings.T
    topk_vals, topk_idx = sim_matrix.topk(top_k + 1, dim=-1)  # +1 because self-similarity included

    edges_src = torch.arange(embeddings.size(0), device=embeddings.device).unsqueeze(1).repeat(1, top_k).flatten()
    edges_dst = topk_idx[:, 1:].flatten()  # skip self-loop
    edge_index = torch.stack([edges_src, edges_dst], dim=0)

    return edge_index

class GATPatchCluster(nn.Module):
    def __init__(self, device, embed_dim=128, gat_heads=4, top_k=10):
        super().__init__()

        # CNN Backbone (ResNet18)
        backbone = models.resnet18(weights='IMAGENET1K_V1')
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])  # [B*N, 512, H', W']
        self.fc = nn.Linear(512, embed_dim)
        self.top_k = top_k
        self.device = device
        # GAT
        self.gat1 = GATConv(embed_dim, embed_dim, heads=gat_heads, concat=False)
        self.gat2 = GATConv(embed_dim, embed_dim, heads=1)

    def forward(self, patches, edge_index):
        feats = self.encoder(patches)  # [B*N, 512, H', W']
        feats = feats.mean(dim=[2, 3])  # Global average pooling [B*N, 512]
        feats = self.fc(feats)  # [B*N, embed_dim]

        x = F.relu(self.gat1(feats, edge_index))
        x = self.gat2(x, edge_index)

        return x  # [B*N, embed_dim]




    def train_model(self, train_loader,val_loader,
              optimizer,scheduler,  device, epochs=10,top_k=10,temp=0.5):


        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for batch, classes in tqdm(train_loader):
                B, N, C, H, W = batch.shape
                batch = batch.view(B * N, C, H, W).to(device)
                labels = torch.arange(B).repeat_interleave(N).to(device)

                with torch.no_grad():
                    initial_embeddings = self.encoder(batch).mean(dim=[2, 3])
                    edge_index = create_patch_graph(initial_embeddings, top_k=top_k).to(device)

                # Forward pass
                embeddings = self.forward(batch, edge_index)

                # Contrastive loss
                loss = supervised_nt_xent_loss(embeddings, labels, temperature=temp)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

            self.eval()
            with torch.no_grad():
                avg_ari = evaluate_clustering_on_validation(val_loader, self,device=device)
                print(f"Epoch [{epoch + 1}/{epochs}], ARI: {avg_ari:.4f}")
                scheduler.step(avg_ari)
                print("current learning rate: ", scheduler.get_last_lr())

    def __call__(self, x):
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor of shape [B*N, C, H, W].
        Returns:
            torch.Tensor: Output tensor of shape [B*N, embed_dim].
        """
        init_embed =self.encoder(x).mean(dim=[2, 3])
        edge_index = create_patch_graph(init_embed, top_k=self.top_k).to(self.device)
        return self.forward(x, edge_index)

        #x = self.fc(x)  # [B*N, embed_dim]


