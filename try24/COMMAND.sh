# 프로젝트 루트에서 (mofs/train_pool_valid, mofs_v2.model 있는 위치)
python main_train.py \
  --data_dir mofs/train_pool_valid \
  --mace_model mofs_v2.model \
  --device cuda \
  \
  --num_episodes 2000 \
  --max_steps 220 \
  --fmax_threshold 0.12 \
  --bond_break_ratio 2.4 \
  --k_bond 3.0 \
  --max_penalty 10.0 \
  \
  --replay_size 300000 \
  --batch_size 1024 \
  --gamma 0.99 \
  --tau 0.005 \
  --n_step 3 \
  \
  --warmup_steps 2000 \
  --updates_per_step 1 \
  \
  --actor_hidden 256 \
  --critic_hidden 256 \
  --init_alpha 0.2 \
  --target_entropy_scale 1.0 \
  \
  --per_alpha 0.6 \
  --per_beta 0.4 \
  \
  --log_dir logs_mof_sac \
  --ckpt_dir ckpt_mof_sac \
  --ckpt_interval 20 \
  --log_interval_steps 50 \
  \
  --seed 42 \
  --cmax 0.1

