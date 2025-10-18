import itertools
import math
import pickle
from collections import defaultdict

import torch
import torch.nn as nn

from rich.console import Console
from rich.table import Table
from tqdm import tqdm


from edl_mirage.architectures.linear_sequential import linear_sequential
from edl_mirage.loss_functions import (
    BeliefMatchingLoss,
    CrossEntropyLoss,
    EnD2Loss,
    UncertainCrossEntropyLoss,
    UnifiedDirichletUQLoss,
    UnifiedSquaredLoss,
)
from edl_mirage.utils.helpers import EarlyStopper, reset_seed
from edl_mirage.utils.metrics import (
    compute_differential_entropy,
    compute_energy,
    compute_max_prob,
    compute_mutual_information,
    compute_precision,
    compute_total_entropy,
)
from edl_mirage.utils.networks import (
    Budget,
    BudgetConstrainedUncertaintyModel,
    ClasswiseBudgetConstrainedUncertaintyModel,
    NaiveUncertaintyModel,
    get_feature_extractor,
    get_flow_model,
    get_onehot_target,
)

"""
This class unified all the classical EDL methods.
"""


class DirichletUQNetwork(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.no_density = configs.no_density
        self.loss_type = configs.loss_type
        reset_seed(configs.seed)
        # Uncertainty model
        # 1-1) linear layers
        if configs.use_local_mlp:
            model_local = linear_sequential(
                input_dims=[configs.latent_dim_local],
                hidden_dims=[configs.hidden_dims[-1]],
                output_dim=configs.input_dim_flow,
                k_lipschitz=configs.k_lipschitz,
            )
        else:
            configs.input_dim_flow = configs.latent_dim_local
            model_local = None
        # 1-2) uncertainty model
        if (
            not configs.no_density
        ):  # for other EDL methods with density parameterization
            budget = Budget(
                configs.budget_function,
                configs.budget_parameter,
                configs.budget_normalized,
            )
            # NatPN uses a single density model
            if configs.single_flow:
                self.uq_model_local = BudgetConstrainedUncertaintyModel(
                    density_estimator=get_flow_model(
                        density_type=configs.density_type,
                        dim=configs.input_dim_flow,
                        n_density=configs.n_density,
                        num_classes=1,
                    ),
                    model_conditional=linear_sequential(
                        input_dims=[configs.latent_dim_local],
                        hidden_dims=[configs.hidden_dims[-1]],
                        output_dim=configs.num_classes,
                        k_lipschitz=configs.k_lipschitz,
                        p_drop=configs.dropout_ratio,
                    ),
                    prior=configs.prior,
                    budget=budget,
                    input_dim_flow=configs.input_dim_flow,
                    model_local=model_local,
                )
            else:  # PostNet uses multiple density models
                self.uq_model_local = ClasswiseBudgetConstrainedUncertaintyModel(
                    density_estimators=get_flow_model(
                        density_type=configs.density_type,
                        dim=configs.input_dim_flow,
                        n_density=configs.n_density,
                        num_classes=configs.num_classes,
                    ),
                    prior=configs.prior,
                    budget=budget,
                    input_dim_flow=configs.input_dim_flow,
                    model_local=model_local,
                )
        else:  # if configs.no_density:
            # if no density, then it's for other EDL methods using direct parameterization
            self.uq_model_local = NaiveUncertaintyModel(
                model_conditional=linear_sequential(
                    input_dims=[configs.latent_dim_local],
                    hidden_dims=[configs.hidden_dims[-1]],
                    output_dim=configs.num_classes,
                    k_lipschitz=configs.k_lipschitz,
                    p_drop=configs.dropout_ratio,
                ),
                model_local=model_local,
            )
        # 3) feature extractor
        self.feature_extractor = get_feature_extractor(
            configs.architecture,
            configs.input_dims,
            configs.hidden_dims,
            configs.kernel_dim,
            configs.latent_dim_local,
            configs.k_lipschitz,
            # configs.dropout_ratio
        )
        # 4) optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=configs.lr)

    def forward(
        self,
        x,
        targets,
        x_ood=None,
        compute_loss=True,
        **kwargs,
    ):
        latent = self.feature_extractor(x)
        alphas = self.uq_model_local(latent)
        soft_output_pred = torch.nn.functional.normalize(alphas, p=1)
        outputs = dict(
            soft=soft_output_pred,
            hard=self.predict(soft_output_pred),
            alphas=alphas,
            latent=latent,
            loss=None,
        )
        if x_ood is not None:
            latent_ood = self.feature_extractor(x_ood)
            alphas_ood = self.uq_model_local(latent_ood)
            outputs["alphas_ood"] = alphas_ood
        if compute_loss:
            assert targets is not None
            outputs["loss"] = self.loss = self.compute_loss(
                outputs["alphas"], targets, **kwargs
            )
            if "alphas_ood" in outputs:
                assert "UnifiedRev" or "MSE" in self.loss_type
                targets_ood = torch.zeros_like(outputs["alphas_ood"])
                self.loss += self.configs.ood_weight * self.compute_loss(
                    outputs["alphas_ood"], targets_ood
                )
                outputs["loss"] = self.loss
        return outputs

    def compute_loss(self, alphas, targets, **kwargs):
        # NOTE: **kwargs is for other arguments, such as temperature argument for EnD^2 loss
        if self.loss_type == "CE":
            loss_ftn = CrossEntropyLoss()
        elif (
            self.loss_type == "MSE"
        ):  # The proposed heuristic MSE loss for ablation study
            loss_ftn = UnifiedSquaredLoss(
                reg_weight=self.configs.reg_weight,
                smooth_epsilon=self.configs.smooth_epsilon,
                prior=self.configs.prior,
            )
        elif self.loss_type == "BM":
            # Belief Matching loss (Being Bayesian about categorical probability; Joo et al., 2019)
            # equivalent to UnifiedDirichletUQLoss(reg_weight=1., prior=1., mode='reverse')
            loss_ftn = BeliefMatchingLoss(
                reg_weight=self.configs.reg_weight, prior=self.configs.prior
            )
        elif self.loss_type == "UCE":
            # Uncertainty Cross Entropy loss (Posterior Networks; Charpentier et al., 2020, 2022)
            # equivalent to UnifiedDirichletUQLoss(reg_weight=1., prior=1., mode='reverse')
            loss_ftn = UncertainCrossEntropyLoss(self.configs.reg_weight)
        elif "Unified" in self.loss_type:
            # ['UnifiedRev', 'UnifiedFwd', 'UnifiedSym']
            mode = (
                "reverse"
                if self.loss_type[-3:] == "Rev"
                else ("forward" if self.loss_type[-3:] == "Fwd" else "symmetric")
            )
            loss_ftn = UnifiedDirichletUQLoss(
                reg_weight=self.configs.reg_weight, prior=self.configs.prior, mode=mode
            )
        elif self.loss_type == "EnD2":
            # For EnD2Loss, targets must be teacher_logits of size (B, M, C)
            loss_ftn = EnD2Loss()
            return loss_ftn(alphas, targets, **kwargs)
        else:
            raise NotImplementedError
        return loss_ftn(alphas, targets)

    def step(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def predict(self, p):
        output_pred = torch.max(p, dim=-1)[1]
        return output_pred


"""
Train function for classical EDL methods.
"""


def train_unified_uq_net(
    configs,
    model_post: DirichletUQNetwork,
    train_loader,
    val_loader,
    device,
    save_dir,
    ood_loader=None,
):
    all_stats = defaultdict(defaultdict(list).copy)
    early_stopper = EarlyStopper(
        patience=configs.early_stop_patience, delta=configs.early_stop_delta
    )
    best_acc = 0.0
    best_loss = math.inf
    for epoch in tqdm(range(configs.num_epoch)):
        loss_total = 0.0
        cnt = 0
        if ood_loader is None:
            for batch_idx, sample in enumerate(train_loader):
                model_post.train()
                input_, target_ = sample
                input_, target_ = input_.to(device), target_.to(device)
                target_ = get_onehot_target(target_, configs.num_classes).to(device)
                cnt += 1
                outputs = model_post(x=input_, targets=target_, compute_loss=True)
                loss_total += outputs["loss"].detach().item()
                model_post.step()
        else:  # If using OOD data for training
            for batch_idx, (sample, sample_ood) in enumerate(
                zip(train_loader, itertools.cycle(ood_loader))
            ):
                model_post.train()
                input_, target_ = sample
                input_, target_ = input_.to(device), target_.to(device)
                input_ood_, *_ = sample_ood
                input_ood_ = input_ood_.to(device)
                target_ = get_onehot_target(target_, configs.num_classes).to(device)
                cnt += 1
                outputs = model_post(
                    x=input_, targets=target_, x_ood=input_ood_, compute_loss=True
                )
                loss_total += outputs["loss"].detach().item()
                model_post.step()

        loss_total /= cnt
        print(f"Epoch {epoch + 1}/{configs.num_epoch}: avg_loss {loss_total}")
        if (epoch + 1) % configs.validation_frequency == 0 or (
            epoch + 1
        ) == configs.num_epoch:
            stats = dict()
            if ood_loader is None:
                stats["train"] = test_unified_uq_net(
                    configs,
                    model_post,
                    train_loader,
                    device,
                )
                stats["val"] = test_unified_uq_net(
                    configs,
                    model_post,
                    val_loader,
                    device,
                )
            else:  # If using OOD data for training
                stats["train"] = test_unified_uq_net_ood(
                    configs,
                    model_post,
                    train_loader,
                    ood_loader,
                    device,
                )
                stats["val"] = test_unified_uq_net_ood(
                    configs,
                    model_post,
                    val_loader,
                    ood_loader,
                    device,
                )

            table = Table(show_header=True, header_style="bold magenta")
            collected_stats = defaultdict(list)
            table.add_column("")
            for train_or_val in stats:
                table.add_column(
                    f"{train_or_val}", style="dim", width=12, justify="right"
                )
                for stat_key in stats[train_or_val]:
                    if stat_key == "num_data":
                        pass
                    else:
                        if stat_key in ["loss", "acc"]:
                            stat = stats[train_or_val][stat_key]
                        else:
                            stat = stats[train_or_val][stat_key].mean()
                        all_stats[train_or_val][stat_key].append(stat)
                        collected_stats[stat_key].append(f"{stat:.4e}")
            console = Console()
            for stat_key in collected_stats:
                table.add_row(stat_key, *collected_stats[stat_key])
            console.print(table)
            with open(f"{save_dir}/stats_seed{configs.seed}.pkl", "wb") as f:
                pickle.dump(all_stats, f)
            # Save the optimal model
            if configs.saving_criterion == "acc":
                if best_acc < stats["val"]["acc"]:
                    best_acc = stats["val"]["acc"]
                    print("Saving Model...")
                    state = {
                        "configs": configs,
                        "model": model_post.state_dict(),
                        "rng_state": torch.get_rng_state(),
                    }
                    torch.save(state, f"{save_dir}/best.pth")
            elif configs.saving_criterion == "loss":
                if best_loss > stats["val"]["loss"]:
                    best_loss = stats["val"]["loss"]
                    print("Saving Model...")
                    state = {
                        "configs": configs,
                        "model": model_post.state_dict(),
                        "rng_state": torch.get_rng_state(),
                    }
                    torch.save(state, f"{save_dir}/best.pth")

        if configs.early_stop and early_stopper(loss_total):
            print("Early stopped!")
            break

    return all_stats


"""
Test function for classical EDL methods without OOD, such as BM, PostNet, Fisher-EDL, etc.
"""


@torch.no_grad()
def test_unified_uq_net(configs, model_post: DirichletUQNetwork, dataloader, device):
    model_post.eval()
    stats = dict(
        loss=0.0,
        acc=0.0,
        mi=[],
        diff_ent=[],
        precs=[],
        energy=[],
        maxp=[],
        ent=[],
        abstain_mask=[],
        num_data=0,
    )
    for input_, target_ in dataloader:
        input_, target_ = input_.to(device), target_.to(device)
        target_ = get_onehot_target(target_, configs.num_classes).to(device)
        outputs_ = model_post(input_, target_, compute_loss=True)
        stats["loss"] += outputs_["loss"].sum().item() * input_.shape[0]
        stats["acc"] += (
            (torch.argmax(outputs_["alphas"], dim=1) == torch.argmax(target_, dim=1))
            .float()
            .sum()
            .item()
        )
        stats["mi"].append(compute_mutual_information(outputs_["alphas"]))
        stats["diff_ent"].append(compute_differential_entropy(outputs_["alphas"]))
        stats["precs"].append(compute_precision(outputs_["alphas"]))
        stats["energy"].append(compute_energy(outputs_["alphas"]))
        stats["maxp"].append(compute_max_prob(outputs_["alphas"]))
        stats["ent"].append(compute_total_entropy(outputs_["alphas"]))
        stats["abstain_mask"].append(
            torch.argmax(outputs_["alphas"], dim=1).ne(torch.argmax(target_, dim=1))
        )
        stats["num_data"] += len(input_)
    for stat_key in stats:
        if stat_key == "num_data":
            pass
        elif stat_key in ["loss", "acc"]:
            stats[stat_key] /= stats["num_data"]
        else:  # stat_key in ['mi', 'diff_ent', 'precs', 'maxp', 'ent', 'abstain_mask']
            stats[stat_key] = torch.cat(stats[stat_key], 0)
            stats[stat_key] = stats[stat_key].data.cpu().numpy()
    return stats


"""
Test function for EDL methods using OOD, such as RPriorNet
"""


@torch.no_grad()
def test_unified_uq_net_ood(
    configs, model_post: DirichletUQNetwork, train_loader, ood_loader, device
):
    model_post.eval()
    stats = dict(
        loss=0.0,
        acc=0.0,
        mi=[],
        mi_ood=[],
        diff_ent=[],
        diff_ent_ood=[],
        precs=[],
        precs_ood=[],
        maxp=[],
        ent=[],
        abstain_mask=[],
        num_data=0,
    )
    for batch_idx, (sample, sample_ood) in enumerate(
        zip(train_loader, itertools.cycle(ood_loader))
    ):
        input_, target_ = sample
        input_, target_ = input_.to(device), target_.to(device)
        input_ood_, *_ = sample_ood
        input_ood_ = input_ood_.to(device)
        target_ = get_onehot_target(target_, configs.num_classes).to(device)

        outputs_ = model_post(
            x=input_, targets=target_, x_ood=input_ood_, compute_loss=True
        )
        stats["loss"] += outputs_["loss"].sum().item() * input_.shape[0]
        stats["acc"] += (
            (torch.argmax(outputs_["alphas"], dim=1) == torch.argmax(target_, dim=1))
            .float()
            .sum()
            .item()
        )
        stats["mi"].append(compute_mutual_information(outputs_["alphas"]))
        stats["mi_ood"].append(compute_mutual_information(outputs_["alphas_ood"]))
        stats["diff_ent"].append(compute_differential_entropy(outputs_["alphas"]))
        stats["diff_ent_ood"].append(
            compute_differential_entropy(outputs_["alphas_ood"])
        )
        stats["precs"].append(compute_precision(outputs_["alphas"]))
        stats["precs_ood"].append(compute_precision(outputs_["alphas_ood"]))
        stats["maxp"].append(compute_max_prob(outputs_["alphas"]))
        stats["ent"].append(compute_total_entropy(outputs_["alphas"]))
        stats["abstain_mask"].append(
            torch.argmax(outputs_["alphas"], dim=1).ne(torch.argmax(target_, dim=1))
        )
        stats["num_data"] += len(input_)
    for stat_key in stats:
        if stat_key == "num_data":
            pass
        elif stat_key in ["loss", "acc"]:
            stats[stat_key] /= stats["num_data"]
        else:  # stat_key in ['mi', 'diff_ent', 'precs', 'maxp', 'ent', 'abstain_mask']
            stats[stat_key] = torch.cat(stats[stat_key], 0)
            stats[stat_key] = stats[stat_key].data.cpu().numpy()
    return stats
