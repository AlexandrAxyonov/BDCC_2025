import torch
import torch.nn.functional as F


def _proto_similarity_matrix(embeddings, prototypes, similarity: str) -> torch.Tensor:
    sim_name = (similarity or "cosine").lower()
    if sim_name in {"euclid", "euclidean", "l2"}:
        sim_name = "inv_euclid"
    if sim_name == "cosine":
        emb_norm = F.normalize(embeddings, dim=1)
        proto_norm = F.normalize(prototypes, dim=1)
        return torch.matmul(emb_norm, proto_norm.t())
    if sim_name == "inv_euclid":
        dist = torch.cdist(embeddings, prototypes, p=2)
        return 1.0 / (1.0 + dist)
    raise ValueError(f"unknown similarity={sim_name!r}")


def prototype_contrastive_loss(
    embeddings,
    labels,
    prototypes,
    num_classes,
    temperature=0.1,
    similarity="cosine",
):
    """
    embeddings: [B, D]
    labels: [B]
    prototypes: [P, D] = [num_classes * n_proto, D]
    """
    device = embeddings.device
    p_total = prototypes.shape[0]
    n_proto = p_total // num_classes

    sim = _proto_similarity_matrix(embeddings, prototypes, similarity) / temperature  # [B, P]

    labels_exp = labels.unsqueeze(1).expand(-1, n_proto)  # [B, n_proto]
    proto_indices = torch.arange(n_proto, device=device).unsqueeze(0)  # [1, n_proto]
    pos_proto_ids = labels_exp * n_proto + proto_indices  # [B, n_proto]

    pos_mask = torch.zeros_like(sim, dtype=torch.bool)  # [B, P]
    pos_mask.scatter_(1, pos_proto_ids, True)

    exp_sim = torch.exp(sim)
    pos_sum = (exp_sim * pos_mask).sum(dim=1)
    all_sum = exp_sim.sum(dim=1)

    loss = -torch.log(pos_sum / all_sum + 1e-8)
    return loss.mean()


def prototype_contrastive_loss_2(
    embeddings,
    labels,
    prototypes,
    num_classes,
    temperature=0.1,
    similarity="cosine",
):
    device = embeddings.device
    batch_size = embeddings.shape[0]
    p_total = prototypes.shape[0]
    n_proto = p_total // num_classes

    sim = _proto_similarity_matrix(embeddings, prototypes, similarity) / temperature  # [B, P]

    proto_indices = torch.arange(n_proto, device=device).unsqueeze(0)
    class_proto_ids = labels.unsqueeze(1) * n_proto + proto_indices  # [B, n_proto]

    sim_in_class = sim.gather(1, class_proto_ids)
    best_in_class = sim_in_class.argmax(dim=1)
    pos_ids = class_proto_ids[torch.arange(batch_size, device=device), best_in_class]

    pos_mask = torch.zeros_like(sim, dtype=torch.bool)
    pos_mask.scatter_(1, pos_ids.unsqueeze(1), True)

    exp_sim = torch.exp(sim)
    pos_sum = (exp_sim * pos_mask).sum(dim=1)
    all_sum = exp_sim.sum(dim=1)

    loss = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8)
    return loss.mean()
