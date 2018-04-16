# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

Training and validation and inference code for ISBI 2018 fundus challenge

### How do I get set up? ###

command usage
evaluation
ex) python evaluation_segmentation.py --pred_dir=../outputs/final_results_segmentation/od_from_fundus_vessel/ --gt_dir=../data/original/OD_Segmentation_Training_Set/


training
od fovea localization
python train_od_fovea_localization.py --gpu_index=1 --batch_size=4 --branching_point=0

python train_od_segmentation.py --gpu_index=0 --batch_size=8

python img_preprocessing.py --n_processe=20 --task=segmentation




vessel extraction 
python extract_vessels.py --img_dir=../data/disc_segmentation/training/images --out_dir=../outputs/vessels_segmentation/training --gpu_index=0

python train_dr_only_features.py --gpu_index=5 --batch_size=32 --loss_type=L2


nohup python train.py --gpu_index=0 --task=SE --batch_size=4 > ../results_outputs/out_SE_unet &

nohup python train.py --gpu_index=1 --task=EX --batch_size=4 > ../results_outputs/out_EX_unet &

nohup python train.py --gpu_index=2 --task=MA --batch_size=4 > ../results_outputs/out_MA_unet &

nohup python train.py --gpu_index=3 --task=HE --batch_size=4 > ../results_outputs/out_HE_unet &

### Who do I talk to? ###

woalsdnd@vuno.co