import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from isaacgymenvs.utils.reformat import omegaconf_to_dict


OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)


def make(
    seed: int, 
    task: str, 
    num_envs: int, 
    sim_device: str,
    rl_device: str,
    graphics_device_id: int = -1,
    headless: bool = False,
    multi_gpu: bool = False,
    virtual_screen_capture: bool = False,
    force_render: bool = True,
    cfg: DictConfig = None,
    dnne_cfg: dict = None
): 
    from isaacgymenvs.utils.rlgames_utils import get_rlgames_env_creator
    # create hydra config if no config passed in
    if cfg is None:
        # reset current hydra config if already parsed (but not passed in here)
        if HydraConfig.initialized():
            task = HydraConfig.get().runtime.choices['task']
            hydra.core.global_hydra.GlobalHydra.instance().clear()

        with initialize(config_path="./cfg"):
            cfg = compose(config_name="config", overrides=[f"task={task}"])
            cfg_dict = omegaconf_to_dict(cfg.task)
            cfg_dict['env']['numEnvs'] = num_envs
    # reuse existing config
    else:
        cfg_dict = omegaconf_to_dict(cfg.task)
    
    # Apply dnne_cfg overrides if provided
    if dnne_cfg is not None:
        # Override physics dt if specified
        if 'physics_dt' in dnne_cfg:
            cfg_dict['sim']['dt'] = dnne_cfg['physics_dt']
        
        # Override sim configuration if specified
        if 'sim' in dnne_cfg:
            # Deep merge sim configuration
            for key, value in dnne_cfg['sim'].items():
                if isinstance(value, dict) and key in cfg_dict['sim']:
                    # Merge nested dictionaries (like physx)
                    if isinstance(cfg_dict['sim'][key], dict):
                        cfg_dict['sim'][key].update(value)
                    else:
                        cfg_dict['sim'][key] = value
                else:
                    cfg_dict['sim'][key] = value
        
        # Users can add other overrides to dnne_cfg as needed

    create_rlgpu_env = get_rlgames_env_creator(
        seed=seed,
        task_config=cfg_dict,
        task_name=cfg_dict["name"],
        sim_device=sim_device,
        rl_device=rl_device,
        graphics_device_id=graphics_device_id,
        headless=headless,
        multi_gpu=multi_gpu,
        virtual_screen_capture=virtual_screen_capture,
        force_render=force_render,
    )
    return create_rlgpu_env()
