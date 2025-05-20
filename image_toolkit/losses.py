import torch
import torch.nn.functional as F


def supervised_nt_xent_loss(embeddings, labels, temperature=0.2):
    embeddings = F.normalize(embeddings, dim=1)
    similarity_matrix = embeddings @ embeddings.T / temperature  # [N, N]

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(embeddings.device)

    # Remove self-contrast cases
    self_mask = torch.eye(mask.shape[0], device=mask.device)
    mask = mask - self_mask

    # Numerically stable logits
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()

    exp_logits = torch.exp(logits) * (1 - self_mask)

    # Log Probabilities
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)

    # Mean of log-over positive pairs
    mask_sum = mask.sum(dim=1)
    # Prevent division by zero
    mask_sum[mask_sum == 0] = 1

    loss = -(mask * log_prob).sum(dim=1) / mask_sum

    return loss.mean()