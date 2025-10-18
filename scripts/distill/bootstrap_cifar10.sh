args=(
  --method bootstrap
  --dataset_name CIFAR10
  --architecture vgg
  --hidden_dims 64 64 64
  --latent_dim_local 6
  --batch_size 64
  # bootstrap configs
  --bootstrap_num_epoch 100
  --bootstrap_early_stop
  --bootstrap_lr 2.5e-4
  --bootstrap_early_stop_patience 10
  --saving_criterion loss
  --validation_frequency 2
)
num_data_list=(50000)
# train 100 bootstrap models
for num_data in "${num_data_list[@]}"; do
  for i in $(seq 0 10 90); do
    bootstrap_seeds=$(seq $i $((i + 9)))
    python scripts/main_distill.py "${args[@]}" --num_data_list $num_data --bootstrap_seeds ${bootstrap_seeds[@]}
  done
done