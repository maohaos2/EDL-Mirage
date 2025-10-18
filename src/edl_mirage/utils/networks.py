import math

import torch
import torch.nn as nn

from edl_mirage.architectures.alexnet_sequential import alexnet
from edl_mirage.architectures.convolution_linear_sequential import (
    convolution_linear_sequential,
)
from edl_mirage.architectures.linear_sequential import linear_sequential
from edl_mirage.architectures.resnet_sequential import resnet18, resnet18_no_batchnorm
from edl_mirage.architectures.vgg_sequential import vgg16, vgg16_bn
from edl_mirage.posterior_networks.batched_normalizing_flow import (
    BatchedNormalizingFlowDensity,
)
from edl_mirage.posterior_networks.mixture_density import MixtureDensity
from edl_mirage.posterior_networks.normalizing_flow import NormalizingFlowDensity
from edl_mirage.utils.helpers import clamp_preserve_gradients


def get_onehot_target(targets, num_classes):
    batch_size = len(targets)
    betas = torch.zeros((batch_size, num_classes))
    betas[torch.arange(batch_size), targets] = 1
    return betas


def get_feature_extractor(
    architecture,
    input_dims,
    hidden_dims,
    kernel_dim,
    latent_dim,
    k_lipschitz,
    p_drop=None,
):
    if architecture == "linear":
        base = linear_sequential(
            input_dims=input_dims,
            hidden_dims=hidden_dims,
            output_dim=latent_dim,
            k_lipschitz=k_lipschitz,
        )
        return base  # For Gaussian data with linear models, do not use batch norm
    elif architecture == "conv":
        assert len(input_dims) == 3
        base = convolution_linear_sequential(
            input_dims=input_dims,
            linear_hidden_dims=hidden_dims,
            conv_hidden_dims=[64, 64, 64],
            output_dim=latent_dim,
            kernel_dim=kernel_dim,
            k_lipschitz=k_lipschitz,
            p_drop=p_drop,
        )
    elif architecture == "vgg":
        assert len(input_dims) == 3
        base = vgg16_bn(output_dim=latent_dim, k_lipschitz=k_lipschitz)
    elif architecture == "vgg_basic":
        assert len(input_dims) == 3
        base = vgg16(output_dim=latent_dim, k_lipschitz=k_lipschitz)
        return base  # do not use batch norm
    elif architecture == "resnet":
        assert len(input_dims) == 3
        base = resnet18(output_dim=latent_dim)
    elif architecture == "resnet_no_batchnorm":
        assert len(input_dims) == 3
        base = resnet18_no_batchnorm(output_dim=latent_dim)
        return base
    elif architecture == "alexnet":
        assert len(input_dims) == 3
        base = alexnet(output_dim=latent_dim)
    else:
        raise NotImplementedError

    return nn.Sequential(base, nn.BatchNorm1d(num_features=latent_dim))


def get_flow_model(
    density_type,
    dim,
    n_density,
    num_classes,
):
    if density_type in ["planar_flow", "radial_flow", "iaf_flow"]:
        density_estimators = nn.ModuleList(
            [
                NormalizingFlowDensity(
                    dim=dim, flow_length=n_density, flow_type=density_type
                )
                for c in range(num_classes)
            ]
        )
    elif density_type == "batched_radial_flow":
        density_estimators = BatchedNormalizingFlowDensity(
            c=num_classes,
            dim=dim,
            flow_length=n_density,
            flow_type=density_type.replace("batched_", ""),
        )
    elif density_type == "normal_mixture":
        density_estimators = nn.ModuleList(
            [
                MixtureDensity(
                    dim=dim, n_components=n_density, mixture_type=density_type
                )
                for c in range(num_classes)
            ]
        )
    else:
        raise NotImplementedError
    if isinstance(density_estimators, nn.ModuleList) and len(density_estimators) == 1:
        return density_estimators[0]
    else:
        return density_estimators


class Budget(nn.Module):
    __BUDGET_FUNCTIONS__ = {
        "id": lambda ns: ns,
        "log": lambda ns: torch.log(ns + 1.0),
        "exp": lambda ns: torch.exp(ns),
        "one": lambda ns: torch.ones_like(ns),
        "param": lambda ns: (
            nn.Parameter(torch.ones_like(ns).float())
            if isinstance(ns, torch.Tensor)
            else nn.Parameter(torch.ones(1))
        ),
    }

    def __init__(
        self,
        budget_function,
        budget_parameter,
        normalize=False,
    ):
        super().__init__()
        self.budget = self.__BUDGET_FUNCTIONS__[budget_function](budget_parameter)
        self.budget_function = budget_function
        self.normalize = normalize

    def forward(self):
        budget = self.budget
        if self.normalize:
            budget = budget / budget.sum()
        return budget


class ClasswiseBudgetConstrainedUncertaintyModel(nn.Module):
    def __init__(
        self,
        density_estimators,
        prior,
        budget,
        input_dim_flow,
        model_local=None,
    ):
        super().__init__()
        self.density_estimators = density_estimators  # q(x|y)
        self.prior = prior
        self.budget = budget
        self.batch_norm = nn.BatchNorm1d(num_features=input_dim_flow)
        self.model_local = model_local

    def forward(self, latent):
        batch_size = len(latent)
        device = latent.device
        budget = self.budget()
        input_to_flow = latent
        if self.model_local is not None:
            input_to_flow = self.model_local(input_to_flow)
        # class-wise flow models (q(x|y))
        num_classes = len(self.density_estimators)
        log_q_latent = torch.zeros((batch_size, num_classes)).to(device)
        alphas = torch.zeros((batch_size, num_classes)).to(device)  # (B, num_classes)
        if isinstance(self.density_estimators, nn.ModuleList):
            for c in range(num_classes):
                log_q_latent[:, c] = self.density_estimators[c].log_prob(
                    input_to_flow
                )  # log q(x|c)
                alphas[:, c] = self.prior + (
                    budget[c] * torch.exp(log_q_latent[:, c])
                )  # a0 + N[c] * q(x|c)
        else:
            log_q_latent = self.density_estimators.log_prob(input_to_flow)
            alphas = self.prior + (budget[:, None] * torch.exp(log_q_latent)).permute(
                1, 0
            )
        return alphas


class BudgetConstrainedUncertaintyModel(nn.Module):
    def __init__(
        self,
        density_estimator,
        model_conditional,
        prior,
        budget,
        input_dim_flow,
        model_local=None,
    ):
        super().__init__()
        self.density_estimator = density_estimator  # q(x)
        self.model_conditional = model_conditional  # q(y|x)
        self.prior = prior
        self.budget = 0.5 * math.log(4 * math.pi) * input_dim_flow
        self.batch_norm = nn.BatchNorm1d(num_features=input_dim_flow)
        self.model_local = model_local

    def forward(self, latent):
        input_to_flow = latent
        if self.model_local is not None:
            input_to_flow = self.model_local(input_to_flow)
        # a single flow model q(x) + a classifier (q(y|x))
        log_q_latent = self.density_estimator.log_prob(input_to_flow)  # log q(x)
        probs = nn.Softmax(dim=-1)(self.model_conditional(latent))  # q(.|x)
        log_q_latent_update = clamp_preserve_gradients(
            self.budget + log_q_latent, lower=-30.0, upper=30.0
        )
        alphas = self.prior + (
            probs * torch.exp(log_q_latent_update).view(-1, 1)
        )  # a0 + N * q(.|x) * q(x)
        return alphas, log_q_latent  # (B, num_classes)


class NaiveUncertaintyModel(nn.Module):
    def __init__(self, model_conditional, model_local=None):
        super().__init__()
        self.model_conditional = model_conditional
        self.model_local = model_local

    def forward(self, latent):
        if self.model_local is not None:
            latent = self.model_local(latent)
        logits = self.model_conditional(latent)
        alphas = torch.exp(logits)
        return alphas
