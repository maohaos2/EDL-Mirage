args=(
  --method end2
  --dataset_name CIFAR10
  --architecture vgg
  --hidden_dims 64 64 64
  --latent_dim_local 6
  --batch_size 64
  # ensemble configs
  --ensemble_num_epoch 100
  --ensemble_early_stop
  --ensemble_lr 2.5e-4
  # end2 configs
  --end2_num_models 100
  --end2_num_epoch 200
  --end2_early_stop
  --end2_lr 1e-3
  --end2_early_stop_patience 10
  --saving_criterion acc
  --validation_frequency 2
  # Temperature Annealing configs
  --init_temperature 5
  --temperature_decay_epoch 30
  --temperature_decay_length 30
)
num_data_list=(50000)
for seed in {0..4}; do
  for num_data in "${num_data_list[@]}"; do
    python scripts/main_distill.py "${args[@]}" --end2_seed $seed --num_data_list $num_data
  done
done