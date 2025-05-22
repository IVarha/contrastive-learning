import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

def cluster_and_score_p(embeddings: torch.Tensor, true_labels: list[int], B) -> float:
    """
    Cluster the embeddings and compute clustering metrics.
    Args:
        :param embeddings (torch.Tensor): Embeddings to cluster.
        :param true_labels (list[int]): True labels for the data.
        :param B (int): Number of figures.
    """
    embeddings_np = embeddings.detach().cpu().numpy()
    clusterer = AgglomerativeClustering(n_clusters=B)

    pred_labels = clusterer.fit_predict(embeddings_np)
    true_labels = np.array(true_labels)

    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    silhouette = silhouette_score(embeddings_np, pred_labels)

    return ari, nmi, silhouette


def evaluate_clustering_on_validation_p(dataloader, model, device):
    """
    Evaluate clustering performance on validation set.
    Args:
        :param dataloader (DataLoader): DataLoader for validation set.
        :param model (nn.Module): Trained model for generating embeddings
        :param device:
    """
    ari_list, nmi_list, sil_list = [], [], []
    for batch, labels in dataloader:
        batch = batch.to(device)
        with torch.no_grad():
            B, N, C, H, W = batch.shape

            # reshape to [B*N, C, H, W]

            true_labels = labels.numpy().repeat(N)
            # print("true_labels ", true_labels)
            batch = batch.view(B * N, C, H, W)
            idx = torch.randperm(batch.shape[0])
            batch = batch[idx]
            true_labels = true_labels[idx]
            # Get embeddings
            embeddings = model(batch)

            # embeddings = embeddings.view(-1, embeddings.shape[-1])
            # print(embeddings.shape)
            # Compute ARI
            ari, nmi, sil = cluster_and_score_p(embeddings, true_labels, B)
            ari_list.append(ari)
            nmi_list.append(nmi)
            sil_list.append(sil)
            # print(f"Adjusted Rand Index (ARI): {ari:.4f}")

    # Average ARI over all batches
    avg_ari = np.mean(ari_list)
    avg_nmi = np.mean(nmi_list)
    avg_sil = np.mean(sil_list)

    return avg_ari, avg_nmi, avg_sil