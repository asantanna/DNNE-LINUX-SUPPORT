# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import hydra

from omegaconf import DictConfig, OmegaConf
from omegaconf import DictConfig, OmegaConf


def preprocess_train_config(cfg, config_dict):
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """

    train_cfg = config_dict['params']['config']

    train_cfg['device'] = cfg.rl_device

    train_cfg['population_based_training'] = cfg.pbt.enabled
    train_cfg['pbt_idx'] = cfg.pbt.policy_idx if cfg.pbt.enabled else None

    train_cfg['full_experiment_name'] = cfg.get('full_experiment_name')

    print(f'Using rl_device: {cfg.rl_device}')
    print(f'Using sim_device: {cfg.sim_device}')
    print(train_cfg)

    try:
        model_size_multiplier = config_dict['params']['network']['mlp']['model_size_multiplier']
        if model_size_multiplier != 1:
            units = config_dict['params']['network']['mlp']['units']
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}')
    except KeyError:
        pass

    return config_dict


@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):

    import logging
    import os
    from datetime import datetime

    # noinspection PyUnresolvedReferences
    import isaacgym
    from isaacgymenvs.pbt.pbt import PbtAlgoObserver, initial_pbt_check
    from isaacgymenvs.utils.rlgames_utils import multi_gpu_get_rank
    from hydra.utils import to_absolute_path
    from isaacgymenvs.tasks import isaacgym_task_map
    import gym
    from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed

    if cfg.pbt.enabled:
        initial_pbt_check(cfg)

    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv
    from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning import amp_continuous
    from isaacgymenvs.learning import amp_players
    from isaacgymenvs.learning import amp_models
    from isaacgymenvs.learning import amp_network_builder
    import isaacgymenvs


    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    def create_isaacgym_env(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed, 
            cfg.task_name, 
            cfg.task.env.numEnvs, 
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{run_name}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs

    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: create_isaacgym_env(**kwargs),
    })

    ige_env_cls = isaacgym_task_map[cfg.task_name]
    dict_cls = ige_env_cls.dict_obs_cls if hasattr(ige_env_cls, 'dict_obs_cls') and ige_env_cls.dict_obs_cls else False

    if dict_cls:
        
        obs_spec = {}
        actor_net_cfg = cfg.train.params.network
        obs_spec['obs'] = {'names': list(actor_net_cfg.inputs.keys()), 'concat': not actor_net_cfg.name == "complex_net", 'space_name': 'observation_space'}
        if "central_value_config" in cfg.train.params.config:
            critic_net_cfg = cfg.train.params.config.central_value_config.network
            obs_spec['states'] = {'names': list(critic_net_cfg.inputs.keys()), 'concat': not critic_net_cfg.name == "complex_net", 'space_name': 'state_space'}
        
        vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: ComplexObsRLGPUEnv(config_name, num_actors, obs_spec, **kwargs))
    else:

        vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

    rlg_config_dict = omegaconf_to_dict(cfg.train)
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)

    observers = [RLGPUAlgoObserver()]

    if cfg.pbt.enabled:
        pbt_observer = PbtAlgoObserver(cfg)
        observers.append(pbt_observer)

    if cfg.wandb_activate:
        cfg.seed += global_rank
        if global_rank == 0:
            # initialize wandb only once per multi-gpu run
            wandb_observer = WandbAlgoObserver(cfg)
            observers.append(wandb_observer)

    # register new AMP network builder and agent
    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
        model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

        return runner

    # convert CLI arguments into dictionary
    # create runner and set the settings
    runner = build_runner(MultiObserver(observers))
    runner.load(rlg_config_dict)
    runner.reset()

    # dump config dict
    if not cfg.test:
        experiment_dir = os.path.join('runs', cfg.train.params.config.name + 
        '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))

        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))

    # DNNE Profiling: Track first/last step timing
    import time
    import json
    import builtins
    
    # Check if profiling is enabled
    dnne_profiling = cfg.get('dnne_profiling', False) or cfg.get('task', {}).get('dnne_profiling', False)
    
    # Debug print function for consistent logging
    def DNNE_print(message):
        """Print with [DNNE_DEBUG] prefix for easy grep filtering"""
        print(f"[DNNE_DEBUG] {message}")

    if dnne_profiling:
        DNNE_print("DNNE Profiling: Recording step timing...")
        # Set global profiling flag and timing storage
        builtins.DNNE_PROFILING = True
        builtins.DNNE_FIRST_STEP_TIME = None
        builtins.DNNE_LAST_STEP_TIME = None
        builtins.DNNE_TOTAL_ENV_STEPS = 0
        # Record overall start time for total elapsed
        start_time = time.perf_counter()
    
    runner.run({
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint': cfg.checkpoint,
        'sigma': cfg.sigma if cfg.sigma != '' else None
    })
    
    # DNNE Profiling: Save timing data after training
    if dnne_profiling:
        try:
            # Get timing data from builtins
            first_step_time = getattr(builtins, 'DNNE_FIRST_STEP_TIME', None)
            last_step_time = getattr(builtins, 'DNNE_LAST_STEP_TIME', None)
            total_env_steps = getattr(builtins, 'DNNE_TOTAL_ENV_STEPS', 0)
            
            if first_step_time and last_step_time and total_env_steps > 0:
                elapsed_time = last_step_time - first_step_time
                
                step_rate_info = {
                    'step_rate_info': {
                        'first_step_time': first_step_time,
                        'last_step_time': last_step_time,
                        'elapsed_time': elapsed_time,
                        'total_env_steps': total_env_steps,
                        'steps_per_second': total_env_steps / elapsed_time if elapsed_time > 0 else 0
                    }
                }
                
                # Merge with existing C++ timing data if available
                timing_file = '/tmp/isaacgym_cpp_timings.json'
                existing_data = {}
                if os.path.exists(timing_file):
                    with open(timing_file, 'r') as f:
                        existing_data = json.load(f)
                
                existing_data.update(step_rate_info)
                
                with open(timing_file, 'w') as f:
                    json.dump(existing_data, f, indent=2)
                
                DNNE_print(f"DNNE Profiling: Recorded {total_env_steps} steps in {elapsed_time:.3f}s")
                DNNE_print(f"DNNE Profiling: Saved step rate: {step_rate_info['step_rate_info']['steps_per_second']:.1f} steps/sec")
            else:
                DNNE_print(f"DNNE Profiling: No step timing data recorded (first={first_step_time is not None}, last={last_step_time is not None}, steps={total_env_steps})")
        except Exception as e:
            DNNE_print(f"DNNE Profiling: Error saving timing data: {e}")


if __name__ == "__main__":
    launch_rlg_hydra()
