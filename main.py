import numpy as np
import random
import torch
import gym
import argparse
import os
import d4rl
from tqdm import trange
from coolname import generate_slug
import time
import json
import yaml
from log import Logger

import utils
from utils import VideoRecorder
import OSC
from eval import eval_policy
from diffusion.diffusion import Diffusion
from diffusion.model import MLP
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="OSC_TD3")             # Policy name
    parser.add_argument("--env", default="hopper-medium-v0")        # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--save_model", default=False, action="store_true")        # Save model and optimizer parameters
    parser.add_argument('--save_model_final', default=True, action='store_true')
    parser.add_argument('--eval_episodes', default=10, type=int)
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--clip_to_eps', default=False, action='store_true')
    parser.add_argument("--exp_name", default="debug")
    # TD3
    parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)    # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--actor_lr', default=None, type=float)
    # TD3 actor-critic
    parser.add_argument('--actor_hidden_dim', default=256, type=int)
    parser.add_argument('--critic_hidden_dim', default=256, type=int)
    parser.add_argument('--actor_init_w', default=None, type=float)
    parser.add_argument('--critic_init_w', default=None, type=float)
    parser.add_argument('--actor_dropout', default=0.1, type=float)
    # TD3 + BC
    parser.add_argument("--alpha", default=0.4, type=float)
    parser.add_argument("--normalize", default=False, action='store_true')
    # diffusion
    parser.add_argument('--behavior_model_path', default=None, type=str)
    parser.add_argument('--beta', default=0.5, type=float)
    parser.add_argument('--latent_dim', default=None, type=int)
    parser.add_argument('--iwae', default=False, action='store_true')
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument("--beta_schedule", default='vp', type=str)
    parser.add_argument("--T", default=5, type=int)
    # OSC
    parser.add_argument('--lambd', default=1.0, type=float)
    parser.add_argument('--without_Q_norm', default=False, action='store_true')
    parser.add_argument('--lambd_cool', default=False, action='store_true')
    parser.add_argument('--lambd_end', default=0.2, type=float)
    parser.add_argument('--osc_alpha', default=10.0, type=float)
    parser.add_argument('--osc_epsilon', default=0.3, type=float)
    # Antmaze
    parser.add_argument('--antmaze_center_reward', default=0.0, type=float)
    parser.add_argument('--antmaze_no_normalize', default=False, action='store_true')
    # Work dir
    parser.add_argument('--notes', default=None, type=str)
    parser.add_argument('--work_dir', default='tmp', type=str)
    parser.add_argument('--dataset_dir', default='tmp', type=str)
    # Config
    parser.add_argument('--config', default=None, type=str)

    args = parser.parse_args()
    # log config
    if args.config is not None:
        with open(args.config, 'r') as f:
            parser.set_defaults(**yaml.load(f.read(), Loader=yaml.FullLoader))
        args = parser.parse_args()
    
    # Build work dir
    base_dir = './output'
    utils.make_dir(base_dir)
    base_dir = os.path.join(base_dir, args.work_dir)
    utils.make_dir(base_dir)
    args.work_dir = os.path.join(base_dir, args.env)
    utils.make_dir(args.work_dir)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d-%H:%M", ts)
    exp_name = str(args.env) + '-bs' + str(args.batch_size)
    if args.policy == 'OSC_TD3':
        exp_name += '-lamb' + str(args.lambd) + '-b' + \
            str(args.beta) + '-a' + str(args.antmaze_center_reward) + '-lr' + str(args.lr)
    else:
        raise NotImplementedError
    exp_name += '-' + args.exp_name
    if args.notes is not None:
        exp_name = args.notes + '_' + exp_name
    args.work_dir = args.work_dir + '/' + exp_name
    utils.make_dir(args.work_dir)

    args.work_dir = args.work_dir + '/seed' + str(args.seed)
    utils.make_dir(args.work_dir)

    args.model_dir = os.path.join(args.work_dir, 'model')
    utils.make_dir(args.model_dir)
    args.video_dir = os.path.join(args.work_dir, 'video')
    utils.make_dir(args.video_dir)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # utils.snapshot_src('.', os.path.join(args.work_dir, 'src'), '.gitignore')

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    env = gym.make(args.env)
    dataset = env.get_dataset(h5path=args.dataset_dir)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        # TD3
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        # OSC
        "lambd": args.lambd,
        "lr": args.lr,
        "actor_lr": args.actor_lr,
        "without_Q_norm": args.without_Q_norm,
        "num_samples": args.num_samples,
        "iwae": args.iwae,
        "actor_hidden_dim": args.actor_hidden_dim,
        "critic_hidden_dim": args.critic_hidden_dim,
        "actor_dropout": args.actor_dropout,
        "actor_init_w": args.actor_init_w,
        "critic_init_w": args.critic_init_w,

        "osc_alpha": args.osc_alpha,
        "osc_epsilon": args.osc_epsilon,

        # finetune
        # "lambd_cool": args.lambd_cool,
        # "lambd_end": args.lambd_end,
    }

    # Initialize policy
    if args.policy == 'OSC_TD3':

        model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)
        behavior = Diffusion(state_dim=state_dim, action_dim=action_dim, model=model, max_action=max_action,
                               beta_schedule=args.beta_schedule, n_timesteps=args.T,
                               ).to(device)
        behavior.load_state_dict(torch.load(args.behavior_model_path))
        # diffusion.load_state_dict(torch.load(args.diffusion_model_path))
        behavior.eval()
        kwargs['behavior'] = behavior
        
        kwargs['beta'] = args.beta
        policy = OSC.OSC_TD3(**kwargs)

    else:
        raise NotImplementedError

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env, dataset=dataset))
    # replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    print("Dataset size:", replay_buffer.reward.shape[0])
    if 'antmaze' in args.env and args.antmaze_center_reward is not None:
        # Center reward for Ant-Maze
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        replay_buffer.reward = np.where(replay_buffer.reward == 1.0, args.antmaze_center_reward, -1.0)
    if args.normalize and not ('antmaze' in args.env and args.antmaze_no_normalize):
        mean, std = replay_buffer.normalize_states()
    else:
        print("No normalize")
        mean, std = 0, 1
    if args.clip_to_eps:
        replay_buffer.clip_to_eps()

    logger = Logger(args.work_dir, use_tb=True)
    video = VideoRecorder(dir_name=args.video_dir)

    # for t in range(int(args.max_timesteps)):
    for t in trange(int(args.max_timesteps)):
        policy.train(replay_buffer, args.batch_size, logger=logger)

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            eval_episodes = 100 if t + 1 == int(args.max_timesteps) and 'antmaze' in args.env else args.eval_episodes
            d4rl_score = eval_policy(args, t + 1, video, logger, policy, args.env,
                                     args.seed, mean, std, eval_episodes=eval_episodes)
            if args.save_model:
                policy.save(args.model_dir)

    if args.save_model_final:
        policy.save(args.model_dir)

    logger._sw.close()
