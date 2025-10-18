args=(
  --method bootstrap-distill
  --dataset_name CIFAR10
  --architecture vgg
  --hidden_dims 64 64 64
  --latent_dim_local 6
  --batch_size 64
  # bootstrap configs
  --bootstrap_num_epoch 100
  --bootstrap_early_stop
  --bootstrap_lr 2.5e-4
  # bootstrap_distill configs
  --bootstrap_distill_num_models 100
  --bootstrap_distill_num_epoch 200
  --bootstrap_distill_early_stop
  --bootstrap_distill_lr 1e-3
  --bootstrap_distill_early_stop_patience 10
  --saving_criterion acc
  --validation_frequency 2
  # Temperature Annealing configs
  --init_temperature 5
  --temperature_decay_epoch 30
  --temperature_decay_length 30
)
num_data_list=(50000)
for num_data in "${num_data_list[@]}"; do
  python scripts/eval_real_data_distill.py "${args[@]}" --num_data_list $num_data --seed_list 0 1 2 3 4
done