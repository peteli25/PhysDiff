python main.py --dataset NIPS_TS_creditcard --batch_size 64 --mafd_components 6 --lambda_aspe 0.35 --q 0.005
python main.py --dataset NIPS_TS_Water --batch_size 64 --mafd_components 6 --lambda_aspe 0.35 --q 0.005
python main.py --dataset NIPS_TS_Swan --batch_size 64 --mafd_components 6 --lambda_aspe 0.35 --q 0.005 
python main.py --dataset PSM --batch_size 64 --mafd_components 6 --window_size 64 --lambda_aspe 0.35 --q 0.005
python main.py --dataset SWaT --batch_size 64 --mafd_components 6 --window_size 64 --q 0.01
python main.py --dataset SMAP --batch_size 64 --mafd_components 6 --window_size 64 --lambda_aspe 0.35 --q 0.005
python main.py --dataset MSL --batch_size 64 --mafd_components 6 --window_size 64 --q 0.005
python main.py --dataset SMD --batch_size 8 --mafd_components 12 --window_size 64 --q 0.005