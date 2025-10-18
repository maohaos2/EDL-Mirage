import argparse
import csv
import os
from distutils.util import strtobool
from types import SimpleNamespace

import torch

from edl_mirage.unified_uq_network import (
    DirichletUQNetwork,
    test_unified_uq_net,
    test_unified_uq_net_ood,
)
from edl_mirage.utils.data import DataProcessing, OOD_DataProcessing
from edl_mirage.utils.helpers import (
    eval_ood_detection,
    eval_selective_classification,
    plot_histogram,
)
from edl_mirage.utils.networks import Budget

torch.set_printoptions(linewidth=200)

parser = argparse.ArgumentParser(description="Classical EDL methods")
parser.add_argument("--root_path", type=str, default="./data/")
# prior means direct parameterization, posterior means density parameterization
parser.add_argument("--method", type=str, choices=["prior", "posterior"])
parser.add_argument(
    "--seed_list",
    type=int,
    nargs="*",
)
parser.add_argument(
    "--dataset_name", default="CIFAR10", type=str, help="dataset to run"
)
# optimization configs
parser.add_argument(
    "--num_epoch", type=int, default=100, help="number of epochs for model training"
)
parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
parser.add_argument(
    "--num_data_list", type=int, nargs="*", help="the list of number of training data"
)
parser.add_argument("--early_stop", action="store_true")
parser.add_argument("--early_stop_delta", type=float, default=0.0)
parser.add_argument("--early_stop_patience", type=int, default=10)
parser.add_argument("--validation_frequency", type=int, default=2)
parser.add_argument(
    "--saving_criterion", default="loss", type=str, choices=["acc", "loss"]
)
# common configs
parser.add_argument(
    "--architecture",
    type=str,
    default="conv",
    choices=["linear", "conv", "vgg", "vgg_basic", "resnet", "resnet_no_batchnorm"],
)
parser.add_argument("--hidden_dims", type=int, nargs="*")
parser.add_argument(
    "--k_lipschitz",
    type=float,
    default=None,
    help="Lipschitz constant. float or None (if no lipschitz)",
)
parser.add_argument("--dropout_ratio", type=float, default=None)
parser.add_argument("--latent_dim_local", type=int, help="for hybrid/posterior network")
parser.add_argument("--use_local_mlp", type=strtobool, default=False)
parser.add_argument("--budget_normalized", type=strtobool, default=False)
parser.add_argument(
    "--budget_function",
    type=str,
    default="id",
    choices=list(Budget.__BUDGET_FUNCTIONS__.keys()),
    help="budget function name applied on class count",
)
parser.add_argument("--prior", type=float, default=1.0)
# flow model configs
parser.add_argument("--single_flow", action="store_true", default=False)
parser.add_argument("--density_type", type=str, default="radial_flow")
parser.add_argument(
    "--n_density", type=int, default=6, help="number of density components"
)
parser.add_argument("--input_dim_flow", type=int, default=6)
parser.add_argument(
    "--no_density", type=strtobool, default=False, help="use density estimator or not"
)
# UnifiedUQNet
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument(
    "--loss_type",
    type=str,
    default="UCE",
    choices=["BM", "MSE", "UCE", "UnifiedRev", "CE"],
)
parser.add_argument("--reg_weight", type=float, default=1e-5)
parser.add_argument(
    "--reg_weight_fisher",
    type=float,
    default=5e-2,
    help="reg weight for fisher EDL loss",
)
parser.add_argument("--smooth_epsilon", type=float, default=0)
parser.add_argument("--ood_weight", type=float, default=5)
parser.add_argument("--use_ood", type=strtobool, default=False)
parser.add_argument("--ood_data", type=str, default="SVHN")
# Histograms
parser.add_argument(
    "--num_bins", type=int, default=100, help="number of bins for plotting the hitogram"
)
args = parser.parse_args()

# Define the hyper-parameter dictionary
configs_dict = {
    "input_dims": [32, 32, 3],  # Input dimension. list of ints
    "kernel_dim": 5,  # Kernel dimension if conv architecture. int
}
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root_path = f"./results/{args.dataset_name}/"
    # Deifne the ood data for each dataset
    if args.dataset_name == "CIFAR10":
        ood_datasets = ["SVHN", "FashionMNIST", "TinyImageNet", "CIFAR10_noise"]
    elif args.dataset_name == "CIFAR100":
        ood_datasets = ["SVHN", "FashionMNIST", "TinyImageNet", "CIFAR100_noise"]
    else:
        raise NotImplementedError
    data_processing = DataProcessing(
        args.dataset_name, root_path=args.root_path, batch_size=args.batch_size
    )
    args.num_classes = 100 if args.dataset_name == "CIFAR100" else 10
    for seed in args.seed_list:
        configs_dict["seed"] = seed
        for num_data in args.num_data_list:
            print(f"Training with {num_data} samples")
            train_loader, val_loader, test_loader = data_processing.get_dataloader(
                num_data
            )
            # Get all labels of training data
            y_train_all = []
            for batch_idx, (X_train, y_train) in enumerate(train_loader):
                y_train_all.append(y_train)
            y_train_all = torch.cat(y_train_all, 0)
            configs_dict["num_data"] = num_data
            if args.single_flow:
                configs_dict["budget_parameter"] = num_data
            else:
                # budget_parameter = count of data from each class in training set. list of ints
                _, budget_parameter = y_train_all.unique(return_counts=True)
                configs_dict["budget_parameter"] = budget_parameter
            configs = SimpleNamespace(**configs_dict, **vars(args))
            print(configs)
            # Define model and train
            if configs.method == "prior":
                print(
                    "Warning: If method is 'prior', automatically set: no_density <- True"
                )
                configs.no_density = True
                if configs.use_ood:
                    if configs.loss_type == "MSE":
                        print("Warning: use heuristic MSE loss to train the model")
                    elif configs.loss_type == "CE":
                        print("Warning: use cross entropy loss to train the model")
                    else:
                        print(
                            "Warning: If method is 'prior' and using ood, automatically set: loss_type <- 'UnifiedRev'"
                        )
                        configs.loss_type = "UnifiedRev"
                else:
                    if configs.loss_type == "MSE":
                        print("Warning: use heuristic MSE loss to train the model")
                    elif configs.loss_type == "CE":
                        print("Warning: use cross entropy loss to train the model")
                    elif configs.loss_type == "UnifiedRev":
                        print("Warning: use unified reverse kl loss to train the model")
                    else:
                        print(
                            "Warning: If method is 'prior' and not using ood, automatically set: loss_type <- 'BM'"
                        )
                        configs.loss_type = "BM"

            model = DirichletUQNetwork(configs).to(device)
            save_dir = os.path.join(
                root_path,
                f"{configs.method}_{configs.dataset_name}_{configs.architecture}"
                f"_{configs.loss_type}"
                f"{f'_epsilon={str(configs.smooth_epsilon)}' if configs.loss_type == 'MSE' else ''}"
                f"{'_sf' if configs.single_flow else ''}"
                f"{f'_latent_dim={str(configs.latent_dim_local)}' if configs.method == 'posterior' else ''}"
                f"{f'_ood={args.ood_data}' if configs.use_ood else '_ood=0'}"
                f"_{str(configs.reg_weight)}"
                f"_ep{configs.num_epoch}_lr{configs.lr}"
                f"{f'_es{configs.early_stop_patience},{configs.early_stop_delta}' if configs.early_stop else ''}"
                f"_{str(num_data)}"
                f"_{str(seed)}",
            )
            checkpoint = torch.load(f"{save_dir}/best.pth")["model"]
            model.load_state_dict(checkpoint)
            model.eval()

            # Initialize the results file
            csv_filename = os.path.join(save_dir, "uq_downstream.csv")
            columns = [
                "UQ Task",
                "Method",
                "Metric",
                "Test Accuracy",
                "Mutual Information",
                "Differential Entropy",
                "Precision",
                "Max Probability",
                "Entropy",
            ]
            with open(csv_filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(columns)
            """
            UQ Evaluation: Selective Classification
            """
            if configs.use_ood:
                stats = test_unified_uq_net_ood(
                    configs, model, test_loader, test_loader, device
                )
            else:
                stats = test_unified_uq_net(configs, model, test_loader, device)
            AUROC, AUPR = eval_selective_classification(stats)
            with open(csv_filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "Selective Classification",
                        args.method,
                        "AUROC",
                        f"{stats['acc']:.3f}",
                        f"{AUROC['mi']:.3f}",
                        f"{AUROC['diff_ent']:.3f}",
                        f"{AUROC['precs']:.3f}",
                        f"{AUROC['maxp']:.3f}",
                        f"{AUROC['ent']:.3f}",
                    ]
                )
                writer.writerow(
                    [
                        "Selective Classification",
                        args.method,
                        "AUPR",
                        f"{stats['acc']:.3f}",
                        f"{AUPR['mi']:.3f}",
                        f"{AUPR['diff_ent']:.3f}",
                        f"{AUPR['precs']:.3f}",
                        f"{AUPR['maxp']:.3f}",
                        f"{AUPR['ent']:.3f}",
                    ]
                )
            """
            UQ Evaluation: OOD Detection
            """
            for ood_dataset in ood_datasets:
                ood_data_processing = OOD_DataProcessing(
                    ood_dataset, root_path=args.root_path, batch_size=args.batch_size
                )
                ood_loader = ood_data_processing.get_dataloader()

                if configs.use_ood:
                    stats_ood = test_unified_uq_net_ood(
                        configs, model, ood_loader, ood_loader, device
                    )
                else:
                    stats_ood = test_unified_uq_net(configs, model, ood_loader, device)
                AUROC, AUPR = eval_ood_detection(stats, stats_ood)
                with open(csv_filename, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [
                            f"OOD Detection: {ood_dataset}",
                            args.method,
                            "AUROC",
                            "/",
                            f"{AUROC['mi']:.3f}",
                            f"{AUROC['diff_ent']:.3f}",
                            f"{AUROC['precs']:.3f}",
                            f"{AUROC['maxp']:.3f}",
                            f"{AUROC['ent']:.3f}",
                        ]
                    )
                    writer.writerow(
                        [
                            f"OOD Detection: {ood_dataset}",
                            args.method,
                            "AUPR",
                            "/",
                            f"{AUPR['mi']:.3f}",
                            f"{AUPR['diff_ent']:.3f}",
                            f"{AUPR['precs']:.3f}",
                            f"{AUPR['maxp']:.3f}",
                            f"{AUPR['ent']:.3f}",
                        ]
                    )
                    # plot histogram
                    histogram_path = os.path.join(save_dir, str(ood_dataset))
                    if not os.path.exists(histogram_path):
                        os.makedirs(histogram_path)
                    plot_histogram(
                        stats, stats_ood, histogram_path, num_bins=args.num_bins
                    )
