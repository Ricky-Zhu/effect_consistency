python collect_data.py --env Ant-v2
python collect_data.py --env Ant_5leg-v2

python alignexp.py --env Ant-v2 --target_env Ant_5leg-v2 \
--pair_n 20000 --display_gap 1000 --eval_gap 1000 --eval_n 10 --pretrain_i 0 --init_start 0 --start_train \
--eval_type 'mujoco' --seed 100

python alignexp.py --env Ant-v2 --target_env Ant_5leg-v2 \
--pair_n 20000 --display_gap 1000 --eval_gap 1000 --eval_n 10 --pretrain_i 1 --init_start 0 --start_train \
--eval_type 'mujoco' --seed 100
