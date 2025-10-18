import argparse
import csv
import os
from distutils.util import strtobool
from types import SimpleNamespace

import torch

from edl_mirage.bootstrap_distill.distill_train_test import test_distill
from edl_mirage.unified_uq_network import DirichletUQNetwork
from edl_mirage.utils.data import (
    DataProcessing,
    Distill_DataProcessing,
    Distill_OOD_DataProcessing,
)
from edl_mirage.utils.helpers import (
    eval_ood_detection,
    eval_selective_classification,
    plot_histogram,
)

torch.set_printoptions(linewidth=200)

parser = argparse.ArgumentParser(
    description="Distillation based methods (includes bootstrap-distillation)"
)
parser.add_argument("--root_path", type=str, default="./data/")
parser.add_argument(
    "--method", type=str, choices=["ensemble", "end2", "bootstrap", "bootstrap-distill"]
)
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
    "--num_data_list", type=int, nargs="*", help="the list of number of training data"
)
parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
# ensemble configs
parser.add_argument("--ensemble_seeds", type=int, nargs="*")
parser.add_argument(
    "--ensemble_num_epoch",
    type=int,
    default=100,
    help="number of epochs for ensemble training",
)
parser.add_argument("--ensemble_early_stop", action="store_true")
parser.add_argument("--ensemble_early_stop_delta", type=float, default=0.0)
parser.add_argument("--ensemble_early_stop_patience", type=int, default=10)
parser.add_argument("--ensemble_lr", type=float, default=1e-3)
# bootstrap configs
parser.add_argument("--bootstrap_seeds", type=int, nargs="*")
parser.add_argument(
    "--bootstrap_num_epoch",
    type=int,
    default=100,
    help="number of epochs for bootstrap training",
)
parser.add_argument("--bootstrap_early_stop", action="store_true")
parser.add_argument("--bootstrap_early_stop_delta", type=float, default=0.0)
parser.add_argument("--bootstrap_early_stop_patience", type=int, default=10)
parser.add_argument("--bootstrap_lr", type=float, default=1e-3)
# end2 configs
parser.add_argument("--end2_num_models", type=int, default=100)
parser.add_argument("--end2_seed", type=int, default=0)
parser.add_argument(
    "--end2_num_epoch", type=int, default=100, help="number of epochs for ed2 training"
)
parser.add_argument("--end2_early_stop", action="store_true")
parser.add_argument("--end2_early_stop_delta", type=float, default=0.0)
parser.add_argument("--end2_early_stop_patience", type=int, default=10)
parser.add_argument("--end2_lr", type=float, default=1e-3)
# bootstrap-distill configs
parser.add_argument("--bootstrap_distill_num_models", type=int, default=100)
parser.add_argument("--bootstrap_distill_seed", type=int, default=0)
parser.add_argument(
    "--bootstrap_distill_num_epoch",
    type=int,
    default=100,
    help="number of epochs for bootstrap-distill training",
)
parser.add_argument("--bootstrap_distill_early_stop", action="store_true")
parser.add_argument("--bootstrap_distill_early_stop_delta", type=float, default=0.0)
parser.add_argument("--bootstrap_distill_early_stop_patience", type=int, default=10)
parser.add_argument("--bootstrap_distill_lr", type=float, default=1e-3)
parser.add_argument("--init_temperature", type=float, default=10.0)
parser.add_argument("--min_temperature", type=float, default=1.0)
parser.add_argument("--temperature_decay_epoch", type=int)
parser.add_argument("--temperature_decay_length", type=int)
# validation configs
parser.add_argument("--validation_frequency", type=int, default=2)
parser.add_argument(
    "--saving_criterion", default="loss", type=str, choices=["acc", "loss"]
)
# common configs
parser.add_argument("--dropout_ratio", type=float, default=None)
parser.add_argument("--dropout_seed", type=int, default=0)
parser.add_argument(
    "--architecture",
    type=str,
    default="conv",
    choices=["linear", "conv", "vgg", "resnet"],
)
parser.add_argument("--hidden_dims", type=int, nargs="*")
parser.add_argument(
    "--k_lipschitz",
    type=float,
    default=None,
    help="Lipschitz constant. float or None (if no lipschitz)",
)
parser.add_argument("--latent_dim_local", type=int, help="for hybrid/posterior network")
parser.add_argument("--use_local_mlp", type=strtobool, default=False)
# Plot Histograms
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
    if not os.path.exists(root_path):
        os.makedirs(root_path)
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
        for num_data in args.num_data_list:
            train_loader, val_loader, test_loader = data_processing.get_dataloader(
                num_data
            )
            # Get all labels of training data
            y_train_all = []
            for batch_idx, (X_train, y_train) in enumerate(train_loader):
                y_train_all.append(y_train)
            y_train_all = torch.cat(y_train_all, 0)
            configs_dict["num_data"] = num_data
            configs = SimpleNamespace(**configs_dict, **vars(args))
            print(
                "Warning: If method is 'ensemble', 'end2', 'bootstrap', 'bootstrap-distill', automatically set: no_density <- True"
            )
            configs.no_density = True
            print(configs)

            if configs.method == "end2":
                ensemble_dir = os.path.join(
                    root_path,
                    f"ensemble_{configs.dataset_name}_{configs.architecture}"
                    f"_ep{configs.ensemble_num_epoch}_lr{configs.ensemble_lr}"
                    f"{f'_es{configs.ensemble_early_stop_patience},{configs.ensemble_early_stop_delta}' if configs.ensemble_early_stop else ''}"
                    f"_{str(num_data)}",
                )
                print(
                    "Warning: If method is 'end2', automatically set: loss_type <- 'EnD2'"
                )
                configs.loss_type = "EnD2"
                # load ensemble logits
                logits_train = []
                logits_val = []
                logits_test = []
                try:
                    # collect all precomputed logits
                    for ensemble_seed in range(configs.end2_num_models):
                        ensemble_dir_seed = os.path.join(
                            ensemble_dir, f"seed{ensemble_seed}"
                        )
                        logits_train.append(
                            torch.load(
                                ensemble_dir_seed + "/logits_train.pth",
                                map_location="cpu",
                            )
                        )
                        logits_val.append(
                            torch.load(
                                ensemble_dir_seed + "/logits_val.pth",
                                map_location="cpu",
                            )
                        )
                        logits_test.append(
                            torch.load(
                                ensemble_dir_seed + "/logits_test.pth",
                                map_location="cpu",
                            )
                        )
                    logits_train = torch.stack(logits_train, dim=1)
                    logits_val = torch.stack(logits_val, dim=1)
                    logits_test = torch.stack(logits_test, dim=1)
                except:
                    raise ValueError("Run first to generate precomputed logits with method='ensemble'!")
                train_loader, val_loader, test_loader = Distill_DataProcessing(
                    args.dataset_name,
                    root_path=args.root_path,
                    batch_size=args.batch_size,
                ).get_dataloader(num_data, logits_train, logits_val, logits_test)

                configs.seed = configs.end2_seed
                configs.num_epoch = configs.end2_num_epoch
                configs.early_stop = configs.end2_early_stop
                configs.early_stop_delta = configs.end2_early_stop_delta
                configs.early_stop_patience = configs.end2_early_stop_patience
                configs.lr = configs.end2_lr
                model = DirichletUQNetwork(configs).to(device)
                save_dir = os.path.join(
                    root_path,
                    f"end2_{configs.dataset_name}_{configs.architecture}"
                    f"_ep{configs.end2_num_epoch}_lr{configs.end2_lr}"
                    f"{f'_es{configs.end2_early_stop_patience},{configs.end2_early_stop_delta}' if configs.end2_early_stop else ''}"
                    f"_{str(num_data)}"
                    f"_seed{seed}",
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
                stats = test_distill(configs, model, test_loader, device)
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
                    ood_data_processing = Distill_OOD_DataProcessing(
                        ood_dataset,
                        root_path=args.root_path,
                        batch_size=args.batch_size,
                    )
                    ood_loader = ood_data_processing.get_dataloader(logits_test)

                    stats_ood = test_distill(configs, model, ood_loader, device)
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
            elif configs.method == "bootstrap-distill":
                bootstrap_dir = os.path.join(
                    root_path,
                    f"bootstrap_{configs.dataset_name}_{configs.architecture}"
                    f"_ep{configs.bootstrap_num_epoch}_lr{configs.bootstrap_lr}"
                    f"{f'_es{configs.bootstrap_early_stop_patience},{configs.bootstrap_early_stop_delta}' if configs.bootstrap_early_stop else ''}"
                    f"_{str(num_data)}",
                )
                print(
                    "Warning: If method is 'bootstrap-distill', automatically set: loss_type <- 'EnD2'"
                )
                configs.loss_type = "EnD2"
                # load ensemble logits
                logits_train = []
                logits_val = []
                logits_test = []
                try:
                    # collect all precomputed logits
                    for bootstrap_seed in range(configs.bootstrap_distill_num_models):
                        bootstrap_dir_seed = os.path.join(
                            bootstrap_dir, f"seed{bootstrap_seed}"
                        )
                        logits_train.append(
                            torch.load(
                                bootstrap_dir_seed + "/logits_train.pth",
                                map_location="cpu",
                            )
                        )
                        logits_val.append(
                            torch.load(
                                bootstrap_dir_seed + "/logits_val.pth",
                                map_location="cpu",
                            )
                        )
                        logits_test.append(
                            torch.load(
                                bootstrap_dir_seed + "/logits_test.pth",
                                map_location="cpu",
                            )
                        )
                    logits_train = torch.stack(logits_train, dim=1)
                    logits_val = torch.stack(logits_val, dim=1)
                    logits_test = torch.stack(logits_test, dim=1)
                except:
                    raise ValueError("Run first to generate precomputed logits with method='bootstrap'!")
                train_loader, val_loader, test_loader = Distill_DataProcessing(
                    args.dataset_name,
                    root_path=args.root_path,
                    batch_size=args.batch_size,
                ).get_dataloader(num_data, logits_train, logits_val, logits_test)

                configs.seed = configs.bootstrap_distill_seed
                configs.num_epoch = configs.bootstrap_distill_num_epoch
                configs.early_stop = configs.bootstrap_distill_early_stop
                configs.early_stop_delta = configs.bootstrap_distill_early_stop_delta
                configs.early_stop_patience = (
                    configs.bootstrap_distill_early_stop_patience
                )
                configs.lr = configs.bootstrap_distill_lr
                model = DirichletUQNetwork(configs).to(device)
                save_dir = os.path.join(
                    root_path,
                    f"bootstrap_distill_{configs.dataset_name}_{configs.architecture}"
                    f"_num_models{str(configs.bootstrap_distill_num_models)}"
                    f"_ep{configs.bootstrap_distill_num_epoch}_lr{configs.bootstrap_distill_lr}"
                    f"{f'_es{configs.bootstrap_distill_early_stop_patience},{configs.bootstrap_distill_early_stop_delta}' if configs.bootstrap_distill_early_stop else ''}"
                    f"_{str(num_data)}"
                    f"_seed{seed}",
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
                stats = test_distill(configs, model, test_loader, device)
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
                    ood_data_processing = Distill_OOD_DataProcessing(
                        ood_dataset,
                        root_path=args.root_path,
                        batch_size=args.batch_size,
                    )
                    ood_loader = ood_data_processing.get_dataloader(logits_test)

                    stats_ood = test_distill(configs, model, ood_loader, device)
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
