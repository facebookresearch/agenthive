import torch
import torchvision.io
from omegaconf import OmegaConf
from torchrl.data import TensorDict
from torchrl.envs import default_info_dict_reader
from torchrl.envs.utils import set_exploration_mode

from rlhive.sim_algos.run import make_env_constructor, make_model


def load_model(model_path, device="cpu", **kwargs):
    # load model and env
    trainer = torch.load(model_path + "trainer.pt", map_location=device)
    env_state_dict = trainer["env"]["worker0"]["env_state_dict"]
    if "worker0" in env_state_dict:
        env_state_dict = env_state_dict["worker0"]
    loss_module_sd = TensorDict(trainer["loss_module"], []).unflatten_keys(".")
    actor_params = loss_module_sd["_actor_network_params"]

    # load config
    yaml_config = torch.load(model_path + "cfg.pt")
    cfg = OmegaConf.create(yaml_config)
    for key, value in kwargs.items():
        cfg.key = value

    # create env
    from_pixels = kwargs.pop("from_pixels", cfg.from_pixels)
    single_env_constructor, _ = make_env_constructor(cfg, stats=None)
    env = single_env_constructor()
    env.set_from_pixels(from_pixels)
    env.load_state_dict(env_state_dict)

    # create policy
    model = make_model(cfg, device, single_env_constructor)
    policy = model[0]
    for i, p in enumerate(policy.parameters()):
        p.data.copy_(actor_params[str(i)].data)

    reader = default_info_dict_reader(["solved"])
    env.set_info_dict_reader(info_dict_reader=reader)
    return env, policy


def rollout_to_video(
    rollout,
    filename: str,
    key="pixels",
):
    """Converts the pixels to a video.
    Pixels should be of shape [T, NUM_CAM, W, H, 3]

    Args:
        rollout (TensorDict): A tensordict containing the rollout.
        filename (str): name of the file where the video will be written.
        key (str, optional): key in the TensorDict where to find the pixel values.
            Default is "pixels"

    """
    pixels = rollout.get(key)
    if not pixels.dtype is torch.uint8:
        raise TypeError(
            f"Expected the pixels tensor to be of type torch.uint8 but got {pixels.dtype}"
        )
    if not pixels.shape[-1] == 3:
        raise RuntimeError(
            f"expected last dimension to be 3 (channels) but got {pixels.shape[-1]}"
        )
    pixels = pixels.permute(0, 1, 4, 2, 3)
    grid = torch.stack(
        [torchvision.utils.make_grid(pixels[t]) for t in range(rollout.shape[0])], 0
    )
    grid = grid.permute(
        0,
        2,
        3,
        1,
    )
    if grid.ndimension() != 4 or grid.shape[-1] != 3:
        raise RuntimeError(f"wrong shape for grid: {grid.shape}")
    torchvision.io.write_video(
        filename,
        grid,
        fps=6,
    )  # video_codec="h264")


if __name__ == "__main__":
    model_path = "/Users/vmoens/Repos/RL/RLHive/rlhive/sim_algos/saved_models/SAC__1b51a3c7_22_09_02-08_08_03/"
    env, policy = load_model(model_path, from_pixels=True)

    with set_exploration_mode("mode"):
        for k in range(10):
            rollout = env.rollout(100, policy=policy)
            print("reward", rollout["reward"].mean())
            print("solved", rollout["solved"].sum())
            rollout["pixels"] = rollout["pixels"].to(torch.uint8)
            rollout_to_video(rollout, f"/Users/vmoens/Desktop/video_{k}.mpg")
