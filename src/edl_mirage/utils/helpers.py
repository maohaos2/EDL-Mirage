import os

import numpy as np
import scipy.ndimage.filters as filters
import torch
from matplotlib import pyplot as plt
from sklearn import metrics

"""
For deterministic training
"""


def reset_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


"""
For data processing: generate corrupted in-distribution data
"""


class PermutationNoise(object):
    def __init__(self):
        pass

    def __call__(self, data):
        shape = data.shape
        new_data = 0 * data
        idx = [torch.tensor(np.random.permutation(np.prod(shape[-2:])))]
        for i, x in enumerate(data):
            new_data[i] = (x.view(np.prod(shape[-2:]))[idx]).view(shape[-2:])
        return new_data


class GaussianFilter(object):
    def __init__(self):
        pass

    def __call__(self, data):
        sigma = 1.0 + 1.5 * torch.rand(1).item()
        return torch.tensor(filters.gaussian_filter(data, sigma, mode="reflect"))


class ContrastRescaling(object):
    def __init__(self):
        pass

    def __call__(self, data):
        gamma = 5 + 25.0 * torch.rand(1).item()
        return torch.sigmoid(gamma * (data - 0.5))


def convert_to_rgb(x):
    return x.convert("RGB")


"""
For stable training: avoid large value
"""


def clamp_preserve_gradients(
    x: torch.Tensor, lower: float, upper: float
) -> torch.Tensor:
    """
    Clamps the values of the tensor into ``[lower, upper]`` but keeps the gradients.

    Args:
        x: The tensor whose values to constrain.
        lower: The lower limit for the values.
        upper: The upper limit for the values.

    Returns:
        The clamped tensor.
    """
    return x + (x.clamp(min=lower, max=upper) - x).detach()


class EarlyStopper:
    def __init__(self, patience=1, delta=0.01, relative=False):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.min_monitor_value = float("inf")
        self.relative = relative

    def __call__(self, monitor_value):
        if monitor_value < self.min_monitor_value:
            self.min_monitor_value = monitor_value
            self.counter = 0
        else:
            # assume self.min_monitor_value > 0.
            diff = monitor_value - self.min_monitor_value
            if self.relative:
                diff /= monitor_value
            if diff > self.delta:
                self.counter += 1
                print(
                    f"Early stopping counter: {self.counter}/{self.patience} (min={self.min_monitor_value})"
                )
                if self.counter >= self.patience:
                    return True
        return False


def eval_ood_detection(stats_test, stats_ood):
    AUROC = dict(mi=0, diff_ent=0, precs=0, maxp=0, ent=0)
    AUPR = dict(mi=0, diff_ent=0, precs=0, maxp=0, ent=0)
    abstain_mask = np.array([0] * stats_test["num_data"] + [1] * stats_ood["num_data"])
    for key in AUROC:
        AUROC[key] = metrics.roc_auc_score(
            abstain_mask, np.concatenate((stats_test[key], stats_ood[key]))
        )
        AUPR[key] = metrics.average_precision_score(
            abstain_mask, np.concatenate((stats_test[key], stats_ood[key]))
        )

    return AUROC, AUPR


def eval_selective_classification(stats_test):
    AUROC = dict(mi=0, diff_ent=0, precs=0, maxp=0, ent=0)
    AUPR = dict(mi=0, diff_ent=0, precs=0, maxp=0, ent=0)
    abstain_mask = stats_test["abstain_mask"]
    for key in AUROC:
        AUROC[key] = metrics.roc_auc_score(abstain_mask, stats_test[key])
        AUPR[key] = metrics.average_precision_score(abstain_mask, stats_test[key])

    return AUROC, AUPR


def plot_histogram(stats_test, stats_ood, root_path, num_bins=200):
    for key in ["mi", "diff_ent", "precs", "maxp", "ent"]:
        plt.figure()
        test_uncertainty = stats_test[key]
        ood_uncertainty = stats_ood[key]
        # Define the number of bins and the range of values for the histograms
        min_value = min(np.min(test_uncertainty), np.min(ood_uncertainty))
        max_value = max(np.max(test_uncertainty), np.max(ood_uncertainty))
        hist1, bin_edges1 = np.histogram(
            test_uncertainty, bins=num_bins, range=(min_value, max_value)
        )
        hist2, bin_edges2 = np.histogram(
            ood_uncertainty, bins=num_bins, range=(min_value, max_value)
        )
        # Plot histograms
        plt.hist(
            test_uncertainty,
            bins=bin_edges1,
            color="blue",
            alpha=0.5,
            label="In-distribution",
        )
        plt.hist(
            ood_uncertainty, bins=bin_edges2, color="orange", alpha=0.5, label="OOD"
        )
        plt.xlabel(str(key))
        plt.ylabel("num_data")
        plt.legend()
        # Save the figure
        plt.savefig(os.path.join(root_path, str(key) + "_histogram.png"))
