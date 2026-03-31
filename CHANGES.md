# Changes introduced through forking the main model

## Commands used to run the models

### CycleGAN

1. **Create environment**: conda env create -f environment.yml
2. **Activate the environment**: conda activate pytorch-img2img
3. **Download the dataset**: bash ./datasets/download_cyclegan_dataset.sh maps
4. **Train**: python train.py --dataroot ./datasets/maps --name maps_cyclegan_smoke --model cycle_gan --n_epochs 1 --n_epochs_decay 1 --save_epoch_freq 1 --save_latest_freq 500 --max_dataset_size 100 --batch_size 4 --use_wandb
5. **Test**: python test.py --dataroot ./datasets/maps --name maps_cyclegan_smoke --model cycle_gan --epoch latest

### Pix2Pix

1. **Create environment**: conda env create -f environment.yml
2. **Activate the environment**: conda activate pytorch-img2img
3. **Download the dataset**: bash ./datasets/download_pix2pix_dataset.sh facades
4. **Train**: python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA --n_epochs 1 --n_epochs_decay 1 --save_epoch_freq 1 --save_latest_freq 500 --max_dataset_size 100 --batch_size 4 --use_wandb
5. **Test**: python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA --epoch latest
