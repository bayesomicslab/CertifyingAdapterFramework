# write a bash script to run the main.py with different noise scales
python main_train.py --exp_name rs_0.25 --noise_scale 0.25 --epochs 200 --methods rs 
python main_train.py --exp_name rs_0.5 --noise_scale 0.5 --epochs 200 --methods rs
python main_train.py --exp_name rs_1 --noise_scale 1 --epochs 200 --methods rs

python main_train.py --exp_name smoothadv_0.25 --noise_scale 0.25 --epochs 200 --methods smoothadv 
python main_train.py --exp_name smoothadv_0.5 --noise_scale 0.5 --epochs 200 --methods smoothadv
python main_train.py --exp_name smoothadv_1 --noise_scale 1 --epochs 200 --methods smoothadv

