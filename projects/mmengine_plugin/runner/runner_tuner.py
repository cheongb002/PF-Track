import copy
from typing import Dict, Optional, Union

from mmengine.config import Config, ConfigDict
from mmengine.runner import Runner
from ..tune import Tuner

ConfigType = Union[Dict, Config, ConfigDict]


class RunnerTuner(Runner):
    @classmethod
    def from_tuning(
        cls,
        runner_cfg: ConfigType,
        hparam_spec: Dict,
        monitor: str,
        rule: str,
        num_trials: int,
        tuning_iter: Optional[int] = None,
        tuning_epoch: Optional[int] = None,
        report_op: str = 'latest',
        searcher_cfg: Dict = dict(type='RandomSearcher')
    ) -> 'Runner':
        """Build a runner from tuning.
        Args:
            runner_cfg (ConfigType): A config used for building runner. Keys of
                ``runner_cfg`` can see :meth:`__init__`.
            hparam_spec (Dict): A dict of hyper parameters to be tuned.
            monitor (str): The metric name to be monitored.
            rule (Dict): The rule to measure the best metric.
            num_trials (int): The maximum number of trials for tuning.
            tuning_iter (Optional[int]): The maximum iterations for each trial.
                If specified, tuning stops after reaching this limit.
                Default is None, indicating no specific iteration limit.
            tuning_epoch (Optional[int]): The maximum epochs for each trial.
                If specified, tuning stops after reaching this number
                of epochs. Default is None, indicating no epoch limit.
            report_op (str):
                Operation mode for metric reporting. Default is 'latest'.
            searcher_cfg (Dict): Configuration for the searcher.
                Default is `dict(type='RandomSearcher')`.
        Returns:
            Runner: A runner build from ``runner_cfg`` tuned by trials.
        """

        runner_cfg = copy.deepcopy(runner_cfg)
        tuner = Tuner(
            runner_cfg=runner_cfg,
            hparam_spec=hparam_spec,
            monitor=monitor,
            rule=rule,
            num_trials=num_trials,
            tuning_iter=tuning_iter,
            tuning_epoch=tuning_epoch,
            report_op=report_op,
            searcher_cfg=searcher_cfg)
        hparam = tuner.tune()['hparam']
        assert isinstance(hparam, dict), 'hparam should be a dict'
        for k, v in hparam.items():
            Tuner.inject_config(runner_cfg, k, v)
        return cls.from_cfg(runner_cfg)

