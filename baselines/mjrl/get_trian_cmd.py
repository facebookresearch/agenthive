import robohive
import click

DESC="""
Script to render trajectories embeded in the env"
"""

@click.command(help=DESC)
@click.option('-s', '--suite', type=str, help='environment suite to train', default="arms")
@click.option('-l', '--launcher', type=click.Choice(['', None, "local", "slurm"]), default='')
@click.option('-cn', '--config_name', type=str, default=None)
@click.option('-cp', '--config_path', type=str, default='config')
def get_train_cmd(suite, launcher, config_name, config_path):

    # Resolve Suite
    if suite=="multitask_":
        envs = ",".join(robohive.robohive_multitask_suite)
        if config_name==None:
            config_name="hydra_kitchen_config.yaml"

    elif suite=="arms":
        envs = ",".join(robohive.robohive_arm_suite)
        if config_name==None:
            config_name="hydra_arms_config.yaml"

    elif suite=="hands":
        envs = ",".join(robohive.robohive_hand_suite)
        if config_name==None:
            config_name="hydra_hand_config.yaml"

    elif suite=="quads":
        envs = ",".join(robohive.robohive_quad_suite)
        if config_name==None:
            config_name="hydra_quads_config.yaml"

    elif suite=="myobase":
        envs = ",".join(robohive.robohive_myobase_suite)
        if config_name==None:
            config_name="hydra_myo_config.yaml"

    elif suite=="myochallenge":
        envs = ",".join(robohive.robohive_myochal_suite)
        if config_name==None:
            config_name="hydra_myo_config.yaml"

    elif suite=="myodm":
        envs = ",".join(robohive.robohive_myodm_suite)
        if config_name==None:
            config_name="hydra_myo_config.yaml"

    else:
        raise ValueError(f"Unsupported suite:{suite}")


    # Resolve launcher
    if launcher=='' or launcher==None:
        launcher_spec = ''
    else:
        launcher_spec = f"--multirun hydra/output={launcher} hydra/launcher={launcher}"

    # Get final training command
    print(f"To train NPG via mjrl on {suite} suite, run the following command: ")
    print(f"python hydra_mjrl_launcher.py --config-path {config_path} --config-name {config_name} {launcher_spec} env={envs} seed=1,2,3")


if __name__ == '__main__':
    get_train_cmd()
