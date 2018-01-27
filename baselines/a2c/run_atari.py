#!/usr/bin/env python3

from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.a2c.a2c import learn
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy

def train(env_id, num_timesteps, seed, policy, lrschedule, num_env,
          replay_lambda=1, replay_loss=None, ss_rate=1, thetas=None):
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    env = VecFrameStack(make_atari_env(env_id, num_env, seed), 4)
    if replay_loss is not None:
        learn_staged(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1),
          lrschedule=lrschedule, replay_lambda=replay_lambda, ss_rate=ss_rate,
         replay_loss=replay_loss, thetas=thetas)
    else:
        learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule)
    env.close()

def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--replay_lambda', help='Replay regularizer parameter', default=1)
    parser.add_argument('--ss_rate', help='Subsampling rate', default=1)
    parser.add_argument('--replay_loss', help='Replay loss, if any', choices=['L2', 'Distillation'], default=None)
    parser.add_argument('--thetas', help='List of thetas to invert over', nargs='*', default=None)
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, lrschedule=args.lrschedule, num_env=16,
          replay_lambda=args.replay_lambda, ss_rate=args.ss_rate,
          replay_loss=args.replay_loss, thetas=args.thetas)

if __name__ == '__main__':
    main()
