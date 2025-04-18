# python main.py --dataset NIPS_TS_creditcard --batch_size 64 --mafd_components 8
# python main.py --dataset NIPS_TS_Water --batch_size 64 --mafd_components 8
# python main.py --dataset NIPS_TS_Swan --batch_size 64 --mafd_components 8
python main.py --dataset SWaT --batch_size 64 --mafd_components 4 --window_size 64
python main.py --dataset MSL --batch_size 64 --mafd_components 4 --window_size 64
python main.py --dataset SMAP --batch_size 64 --mafd_components 4 --window_size 64
python main.py --dataset SMD --batch_size 64 --mafd_components 4 --window_size 64
python main.py --dataset PSM --batch_size 64 --mafd_components 4 --window_size 64
