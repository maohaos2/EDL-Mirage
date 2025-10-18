import torch


def compute_mutual_information(alphas):
    alpha0 = torch.sum(alphas, 1)
    prob = alphas / alpha0.unsqueeze(-1)
    mi = -torch.sum(
        prob
        * (
            torch.log(prob)
            - torch.digamma(alphas + 1)
            + torch.digamma(alpha0.unsqueeze(-1) + 1)
        ),
        -1,
    )
    return mi


def compute_differential_entropy(alphas):
    alpha0 = torch.sum(alphas, -1)
    diff_ent = (
        torch.sum(torch.lgamma(alphas), -1)
        - torch.lgamma(alpha0)
        - torch.sum(
            (alphas - 1)
            * (torch.digamma(alphas) - torch.digamma(alpha0.unsqueeze(-1))),
            -1,
        )
    )
    return diff_ent


def compute_precision(alphas):
    # Precision for Dirichlet output
    # Note: Negative sign added to be consistent with other uncertainty metric: large value means less confident
    return -torch.sum(alphas, 1)


# For heuristic MSE loss in ablation study
def compute_energy(alphas):
    # Free-energy for Dirichlet output
    # Note: Negative sign added to be consistent with other uncertainty metric: large value means less confident
    return -torch.log(torch.sum(alphas, 1))


def _compute_entropy(input_):
    entropy = -input_ * torch.log(input_ + 1e-10)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def compute_total_entropy(alphas):
    # Standard entropy from Dirichlet output
    alpha0 = torch.sum(alphas, 1)
    return _compute_entropy(alphas / alpha0.unsqueeze(-1))


def compute_max_prob(alphas):
    # Max probability from Dirichlet output
    # Note: Negative sign added to be consistent with other uncertainty metric: large value means less confident
    alpha0 = torch.sum(alphas, 1)
    predictions = alphas / alpha0.unsqueeze(-1)
    confidence, _ = torch.max(predictions, 1)
    return -confidence
