import torch
import torch.nn as nn
from torch.distributions import Dirichlet
from torch.nn import functional as F

"""
Unified KL objective
"""


class UnifiedDirichletUQLoss:
    def __init__(self, reg_weight, prior=1.0, mode="reverse"):
        self.reg_weight = reg_weight
        self.prior = prior
        self.mode = mode
        self.dir_kl_div_ftn = DirichletKLDivergence(mode=mode)

    def __call__(self, alphas, targets):
        return self.dir_kl_div_ftn(
            alphas, self.prior * torch.ones_like(alphas) + 1 / self.reg_weight * targets
        )


class DirichletKLDivergence:
    def __init__(self, mode):
        assert mode in ["reverse", "forward", "symmetric"]
        self.mode = mode

    def __call__(self, alphas_pred, alphas_target):
        if self.mode == "reverse":
            return self._kl_divergence(alphas_pred, alphas_target)
        elif self.mode == "forward":
            return self._kl_divergence(alphas_target, alphas_pred)
        else:
            return self._kl_divergence(
                alphas_pred, alphas_target
            ) + self._kl_divergence(alphas_target, alphas_pred)

    @staticmethod
    def _kl_divergence(alphas, betas):
        """
        Adapted from the original implementation of posterior network (see PosteriorNetwork.RKL_loss)
        Compute KL divergence between two Dirichlet distributions: KL( Dir(\alpha), Dir(\beta) )
        """
        alpha0 = torch.sum(alphas, -1)
        beta0 = torch.sum(betas, -1)
        t1 = torch.lgamma(alpha0) - torch.lgamma(beta0)
        t2 = (torch.lgamma(alphas) - torch.lgamma(betas)).sum(-1)
        t3 = alphas - betas
        t4 = alphas.digamma() - alpha0.digamma().unsqueeze(-1)
        loss = t1 - t2 + (t3 * t4).sum(-1)
        return loss.mean()


"""
Standard Cross-entropy loss
"""


class CrossEntropyLoss:
    def __call__(self, alphas, targets):
        pred = torch.nn.functional.normalize(alphas, p=1)
        loss = -torch.sum(targets.squeeze() * torch.log(pred + 1e-8)) / len(alphas)
        return loss


"""
The proposed heuristic MSE loss used in ablation study
"""


class UnifiedSquaredLoss:
    def __init__(self, reg_weight, smooth_epsilon, prior=1.0):
        self.reg_weight = reg_weight
        self.prior = prior
        self.epsilon = smooth_epsilon
        self.SquaredLoss = SquaredLoss()

    def __call__(self, alphas, targets):
        smooth_targets = (1 - self.epsilon) * targets + (
            self.epsilon / targets.size()[1]
        )
        return self.SquaredLoss(
            alphas,
            self.prior * torch.ones_like(alphas) + 1 / self.reg_weight * smooth_targets,
        )


class SquaredLoss:
    def __call__(self, alphas, targets):
        loss = ((torch.log(alphas) - torch.log(targets)) ** 2).sum(-1)
        return torch.mean(loss)


"""
The VI (or BM) loss. Reference: https://github.com/tjoo512/belief-matching-framework/blob/master/loss.py 
"""


class BeliefMatchingLoss(nn.Module):
    def __init__(self, reg_weight, prior=1.0):
        super(BeliefMatchingLoss, self).__init__()
        self.prior = prior
        self.reg_weight = reg_weight

    def forward(self, alphas, targets):
        """
        Compute original Bayesian loss: - log_likelihood + coeff * kl

        targets: onehot-encoded labels
        """

        # compute log-likelihood loss: psi(alpha_zero) - psi(alpha_target)
        alpha0 = alphas.sum(1)
        ll_loss = (
            torch.digamma(alpha0) - (targets * torch.digamma(alphas)).sum(-1)
        ).mean()
        # compute KL loss D(q(\pi|x) || p(\pi))
        # note: equivalent to - Dirichlet(alphas).entropy() up to an additive constant if self.prior = 1.
        kl_loss = DirichletKLDivergence(mode="reverse")(
            alphas, self.prior * torch.ones_like(alphas)
        )
        return ll_loss + self.reg_weight * kl_loss


"""
The UCE  loss. Reference: https://github.com/sharpenb/Posterior-Network/blob/main/src/posterior_networks/PosteriorNetwork.py
"""


class UncertainCrossEntropyLoss:
    def __init__(self, reg_weight):
        self.reg_weight = reg_weight

    def __call__(self, alphas, targets):
        """
        targets: one-hot encoded
        """
        # equivalent to BeliefMatchingLoss(reg_weight=self.reg_weight, prior=1.)(alphas, targets) up to an additive constant
        alpha0 = alphas.sum(1)
        ll_loss = (
            torch.digamma(alpha0) - (targets * torch.digamma(alphas)).sum(-1)
        ).mean()
        kl_loss = -Dirichlet(alphas).entropy().mean()
        return ll_loss + self.reg_weight * kl_loss


"""
Distillation based loss. Reference: https://github.com/KaosEngineer/PriorNetworks/blob/master/prior_networks/ensembles/losses.py
"""


class EnD2Loss:
    """Standard Negative Log-likelihood of the ensemble predictions"""

    def __init__(self, smoothing=1e-8, teacher_prob_smoothing=1e-3):
        self.smooth_val = smoothing
        self.tp_scaling = 1 - teacher_prob_smoothing

    def __call__(self, alphas, teacher_logits, temperature=1.0):
        """
        alphas: (B, C)  # outputs from the model being trained
        teacher_logits: (B, M, C)  # collected logits from the pretrained ensemble
        T: temperature
        """
        alphas = torch.log(alphas + 1e-8)
        # alphas = alphas ** (1 / T)
        alphas = torch.exp(alphas / temperature)
        alpha0 = torch.sum(alphas, dim=1)

        teacher_logits = torch.log(teacher_logits + 1e-8)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=2)
        # Smooth for num. stability:
        probs_mean = 1 / (teacher_probs.size()[2])
        # Subtract mean, scale down, add mean back)
        teacher_probs = self.tp_scaling * (teacher_probs - probs_mean) + probs_mean
        assert torch.all(teacher_probs != 0).item()

        log_teacher_probs_geo_mean = torch.mean(
            torch.log(teacher_probs + self.smooth_val), dim=1
        )
        assert torch.all(torch.isfinite(log_teacher_probs_geo_mean)).item()

        # Define the cost in two parts (dependent on targets and independent of targets)
        target_independent_term = torch.sum(
            torch.lgamma(alphas + self.smooth_val), dim=1
        ) - torch.lgamma(alpha0 + self.smooth_val)
        assert torch.all(torch.isfinite(target_independent_term)).item()

        target_dependent_term = -torch.sum(
            (alphas - 1.0) * log_teacher_probs_geo_mean, dim=1
        )
        assert torch.all(torch.isfinite(target_dependent_term)).item()

        cost = target_dependent_term + target_independent_term
        assert torch.all(torch.isfinite(cost)).item()

        return torch.mean(cost) * (temperature**2)
