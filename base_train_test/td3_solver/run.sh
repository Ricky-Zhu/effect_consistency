#python train.py --env HalfCheetah-v2 --optimal --max_timesteps 1000000 --discount 0.99
#python train.py --env HalfCheetah_3leg-v2 --optimal --max_timesteps 1000000 --discount 0.99
python train.py --env Swimmer-v2 --optimal --max_timesteps 1000000 --discount 0.95
#python train.py --env Swimmer_4part-v2 --optimal --max_timesteps 1000000 --discount 0.95
#
#python test.py --env HalfCheetah-v2 --optimal
#python test.py --env HalfCheetah_3leg-v2 --optimal
#python test.py --env Swimmer-v2 --optimal
#python test.py --env Swimmer_4part-v2 --optimal
