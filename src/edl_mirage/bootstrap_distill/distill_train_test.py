import math
import pickle
from collections import defaultdict

import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from edl_mirage.unified_uq_network import DirichletUQNetwork
from edl_mirage.utils.helpers import EarlyStopper
from edl_mirage.utils.metrics import (
    compute_differential_entropy,
    compute_max_prob,
    compute_mutual_information,
    compute_precision,
    compute_total_entropy,
)
from edl_mirage.utils.networks import get_onehot_target

"""
For extraction of Logits from ensemble or bootstrap models
"""


@torch.no_grad()
def collect_logits(configs, model: DirichletUQNetwork, train_loader_ordered, device):
    logits = []
    for input_, *_ in train_loader_ordered:
        input_ = input_.to(device)
        outputs_ = model(input_, targets=None, compute_loss=False)
        logits.append(outputs_["alphas"].cpu())
    logits = torch.cat(logits, 0)
    return logits


"""
Temperature Annealing
Reference: https://github.com/KaosEngineer/PriorNetworks/blob/master/prior_networks/ensembles/training.py
"""


class _TempScheduler(object):
    def __init__(self, init_temp, last_epoch=-1):
        if last_epoch == -1:
            last_epoch = 0

        self.temp = init_temp
        self.step(last_epoch)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.temp = self.update_temp()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__
        """
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_temp(self):
        raise NotImplementedError

    def update_temp(self):
        raise NotImplementedError


class LRTempScheduler(_TempScheduler):
    def __init__(
        self, init_temp, decay_epoch, decay_length, min_temp=1.0, last_epoch=-1
    ):
        assert decay_length > 0
        assert decay_epoch > 0
        self.decay_epoch = decay_epoch
        self.decay_length = decay_length
        self.init_temp = init_temp
        self.min_temp = min_temp
        super(LRTempScheduler, self).__init__(init_temp, last_epoch)

    def update_temp(self):
        if self.last_epoch <= self.decay_epoch:
            return self.temp
        elif self.last_epoch >= self.decay_epoch + self.decay_length:
            return self.min_temp
        else:
            slope = (self.init_temp - self.min_temp) / self.decay_length
            return self.init_temp - slope * (self.last_epoch - self.decay_epoch)

    def get_temp(self):
        return self.temp


"""
Training function for distillation based methods
"""


def train_distill(
    configs,
    model_distill: DirichletUQNetwork,
    train_loader,
    val_loader,
    device,
    save_dir,
):
    all_stats = defaultdict(defaultdict(list).copy)
    early_stopper = EarlyStopper(
        patience=configs.early_stop_patience, delta=configs.early_stop_delta
    )
    temperature_scheduler = LRTempScheduler(
        init_temp=configs.init_temperature,
        min_temp=configs.min_temperature,
        decay_epoch=configs.temperature_decay_epoch,
        decay_length=configs.temperature_decay_length,
    )
    best_acc = 0.0
    best_loss = math.inf
    for epoch in tqdm(range(configs.num_epoch)):
        loss_total = 0.0
        cnt = 0
        for batch_idx, sample in enumerate(train_loader):
            model_distill.train()
            input_, *_, teacher_logits_ = sample
            input_, teacher_logits_ = input_.to(device), teacher_logits_.to(device)
            # train the ensemble distillation network
            cnt += 1
            outputs = model_distill(
                x=input_,
                targets=teacher_logits_,
                temperature=temperature_scheduler.get_temp(),
                compute_loss=True,
            )
            loss_total += outputs["loss"].detach().item()
            model_distill.step()

        loss_total /= cnt
        print(f"Epoch {epoch + 1}/{configs.num_epoch}: avg_loss {loss_total}")
        if (
            epoch + 1
        ) % configs.validation_frequency == 0 or epoch + 1 == configs.num_epoch:
            stats = dict()
            stats["train"] = test_distill(
                configs,
                model_distill,
                train_loader,
                device,
            )
            stats["val"] = test_distill(
                configs,
                model_distill,
                val_loader,
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
                        "model": model_distill.state_dict(),
                        "rng_state": torch.get_rng_state(),
                    }
                    torch.save(state, f"{save_dir}/best.pth")
            elif configs.saving_criterion == "loss":
                if best_loss > stats["val"]["loss"]:
                    best_loss = stats["val"]["loss"]
                    print("Saving Model...")
                    state = {
                        "configs": configs,
                        "model": model_distill.state_dict(),
                        "rng_state": torch.get_rng_state(),
                    }
                    torch.save(state, f"{save_dir}/best.pth")
            temperature_scheduler.step()
        if configs.early_stop and early_stopper(loss_total):
            print("Early stopped!")
            break

    return all_stats


"""
Testing function for distillation based methods
"""


@torch.no_grad()
def test_distill(configs, model_distill: DirichletUQNetwork, dataloader, device):
    model_distill.eval()
    stats = dict(
        loss=0.0,
        acc=0.0,
        mi=[],
        diff_ent=[],
        precs=[],
        maxp=[],
        ent=[],
        abstain_mask=[],
        num_data=0,
    )
    for input_, target_, teacher_logits_ in dataloader:
        input_, target_, teacher_logits_ = (
            input_.to(device),
            target_.to(device),
            teacher_logits_.to(device),
        )
        target_ = get_onehot_target(target_, configs.num_classes).to(device)
        outputs_ = model_distill(input_, teacher_logits_, compute_loss=True)
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
