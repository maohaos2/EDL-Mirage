# Are Uncertainty Quantification Capabilities of Evidential Deep Learning a Mirage?
[![Paper](https://img.shields.io/badge/arXiv-paper-b31b1b)](https://proceedings.neurips.cc/paper_files/paper/2024/file/c3177be226ee12e34d6ba3b5e6fe6a5b-Paper-Conference.pdf) &nbsp;&nbsp;


This repository contains the official implementation of the paper [Are Uncertainty Quantification Capabilities of Evidential Deep Learning a Mirage?](https://proceedings.neurips.cc/paper_files/paper/2024/file/c3177be226ee12e34d6ba3b5e6fe6a5b-Paper-Conference.pdf) (NeurIPS 2024)



## Requirements 
Warning: this code only works with PyTorch <=1.13.1.
```bash
conda create -n edl python=3.10 -y
conda activate edl
pip install --index-url https://download.pytorch.org/whl/cu117 \
  torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1
conda install -y intel-openmp
pip install "numpy==1.24.4" "scipy<1.11" "pillow<10" "scikit-learn<1.4"
pip install matplotlib rich tqdm pandas
pip install "pyro-ppl<=1.8.6"

# Install the package
pip install -e .
```
## File Structure
### scripts folder: 
- "Classical" contains the scripts to run classical EDL methods.
- "Distill" contains the scripts to run Distillation based EDL methods, including new proposed Bootstrap-Distill
method. 
- "Analysis" includes the scripts to reproduce the empirical findings we discussed in section 5.
### src folder:
Mainly contains the source code for different base model architectures.
### Other code files:
Code files with "distill" are used for distillation based EDL methods, while code files with "unified" are 
used for classical EDL methods. Code files with "eval" are used for downstream task evaluation. 
## Usage
We provide the scripts to help reproduce the empirical results of this paper.
We provide some examples as follows,
#### To train new proposed Bootstrap-Distill method and evaluate it on downstream tasks.
```
. scripts/distill/bootstrap_cifar10.sh
. scripts/distill/bootstrap_distill_cifar10.sh
. scripts/distill/eval_bootstrap_distill_cifar10.sh
```
Note: Before executing bootstrap_distill_cifar10.sh, make sure to run bootstrap_cifar10.sh first. This step generates a collection of bootstrap models that will be used for distillation.
#### To train classical EDL method such as  Posterior Network and evaluate it on downstream tasks.
```
. scripts/classical/PostNet_cifar10.sh
. scripts/classical/eval_PostNet_cifar10.sh
```
#### To reproduce the results about epistemic uncertainty v.s. num_data in Section 5.1
```
. scripts/analysis/epistemic_num_data.sh
```
#### To reproduce the results about effect of lambda to OOD detection performance in Section 5.2
```
. scripts/analysis/ood_lambda.sh
```
#### To reproduce the ablation study of objective function in Section 5.2
```
. scripts/analysis/ablation_objective.sh
```
#### To reproduce the ablation study of model architecture in Section 5.3
```
. scripts/analysis/ablation_RPriorNet.sh
```
#### To reproduce the ablation study of density model in Section 5.3
```
. scripts/analysis/ablation_PostNet.sh
```

## **Contact Information**
For questions, please:
- Raise an issue in our GitHub repository;
- and contact us:
  - Maohao Shen ([maohao@mit.edu](mailto:maohao@mit.edu))
  - Jongha (Jon) Ryu ([jongha.ryu@gmail.com](mailto:jongha.ryu@gmail.com))

## Reference
This codebase is built upon the following repositories:
- [Posterior Network](https://github.com/sharpenb/Posterior-Network)
- [Prior Network](https://github.com/KaosEngineer/PriorNetworks/)
- [Fisher EDL](https://github.com/danruod/IEDL)

## **Citation**
If you find this repository useful in your research, please cite:
```
@inproceedings{shen2024uncertainty,
  title={Are uncertainty quantification capabilities of evidential deep learning a mirage?},
  author={Shen, Maohao and Ryu, Jongha Jon and Ghosh, Soumya and Bu, Yuheng and Sattigeri, Prasanna and Das, Subhro and Wornell, Gregory},
  booktitle={Advances in Neural Information Processing Systems},
  volume={37},
  pages={107830--107864},
  year={2024}
}
```
