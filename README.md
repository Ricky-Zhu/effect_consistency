## collect data with a random policy from both domains
```shell
# collect source domain data
python collect_data.py --data_id 1 --env HalfCheetah-v2
# collect target domain data
python collect_data.py --data_id 1 --env HalfCheetah_3leg-v2
```

## obtain the policy in the source domain
```shell
cd base_train_test/td3_solver
python train.py 
```

## train the correspondence
```shell
cd effect_cycle_transfer
python alignexp.py --env HalfCheetah-v2 --target_env HalfCheetah_3leg-v2 \
--pair_n 7000 --display_gap 1000 --eval_gap 1000 --pretrain_i True --start_train 
```

for swimmer experiments
```shell
CUDA_VISIBLE_DEVICES=2 python alignexp.py --env Swimmer-v2 --target_env Swimmer_4part-v2 --pair_n 7000 --display_gap 1000 \
--eval_gap 1000 --start_train
```