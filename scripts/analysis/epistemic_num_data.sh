# Being Bayesian
args=(
  --method prior
  --dataset_name CIFAR10
  --architecture vgg
  --hidden_dims 64 64 64
  --num_epoch 300
  --lr 2.5e-4
  --early_stop
  --latent_dim_local 6
  --use_ood 0 # Not use OOD
  --no_density 1 # Not use density estimator
  --prior 1.
  --early_stop_patience 30
  --loss_type UnifiedRev
  --saving_criterion loss
  --validation_frequency 2
)
num_data_list=(1000 2000 5000 10000 20000 50000)
reg_weight_list=("1e-4")
for seed in {0..4}; do
  for reg_weight in "${reg_weight_list[@]}"; do
    for num_data in "${num_data_list[@]}"; do
      python scripts/main_unified.py "${args[@]}" --seed $seed --num_data_list $num_data --reg_weight $reg_weight
    done
  done
done