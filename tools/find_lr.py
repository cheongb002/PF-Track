import argparse
import tempfile

from mmengine.config import Config

from projects.mmengine_plugin.runner import RunnerTuner


def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Tuning')
    parser.add_argument('config', help='tuner config file path')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    runner_cfg = Config.fromfile(args.config)

    runner = RunnerTuner.from_tuning(
        runner_cfg=runner_cfg,
        hparam_spec={
            'optim_wrapper.optimizer.lr': {
                'type': 'continuous',
                'lower': 1e-5,
                'upper': 1e-3
            }
        },
        monitor='train/loss',
        rule='less',
        num_trials=32,
        tuning_epoch=2,
        searcher_cfg=dict(type='NevergradSearcher'),
    )


if __name__ == '__main__':
    main()
