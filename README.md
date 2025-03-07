# Only Support Constraint

## Environment

1. Install [MuJoCo version 2.0](https://www.roboti.us/download.html) at ~/.mujoco/mujoco200 and copy license key to ~/.mujoco/mjkey.txt
2. Create a conda environment
```
conda env create -f environment.yml
conda activate osc
```
3. Install [D4RL](https://github.com/Farama-Foundation/D4RL/tree/4aff6f8c46f62f9a57f79caa9287efefa45b6688)

## Usage
### Behavior Clone

Run the following command to train diffusion.


```
python train_diffusion.py --exp diffusionbc --algo bc --device 0 --env_name walker2d-medium-v2 --dir ./output/train_diffusion --dataset_dir ./d4rl/datasets/gym_mujoco_v2/walker2d_medium-v2.hdf5 --save_best_model
```
#### Logging

This codebase uses viskit(https://github.com/vitchyr/viskit)
 You can view saved runs with:

```
python ./viskit/viskit/frontend.py <run_dir>
```


### Offline RL
Run the following command to train offline RL on D4RL with pretrained diffusion models.

```
python main.py --config ./configs/offline/walker2d-medium.yml --exp_name osc --seed 0
```

#### Logging

This codebase uses tensorboard. You can view saved runs with:

```
tensorboard --logdir <run_dir>
```

### Online Fine-tuning

Run the following command to online fine-tune on AntMaze with pretrained diffusion models and offline models.

```
python main_finetune.py --config configs/online_finetune/antmaze-large-diverse.yml --exp_name osc_finetune --seed 0
```