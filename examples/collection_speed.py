import os

import torch.cuda

os.environ["sim_backend"] = "MUJOCO"

import argparse
import time

import tqdm

from rlhive.rl_envs import RoboHiveEnv
from torchrl.collectors.collectors import MultiaSyncDataCollector, RandomPolicy, SyncDataCollector
from torchrl.envs import (
    EnvCreator,
    ParallelEnv,
    SerialEnv,
    R3MTransform,
    TransformedEnv,
)

parser = argparse.ArgumentParser()
parser.add_argument("--num_workers", default=2, type=int)
parser.add_argument("--num_collectors", default=4, type=int)
parser.add_argument("--frames_per_batch", default=50, type=int)
parser.add_argument("--total_frames", default=20_000, type=int)
parser.add_argument("--r3m", action="store_true")
parser.add_argument("--env_name", default="franka_micro_random-v3")
parser.add_argument("--serial", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    device = "cuda:0" if torch.has_cuda else "cpu"
    if args.serial:
        penv = SerialEnv(
            args.num_workers,
            lambda: RoboHiveEnv(args.env_name, device=device),
        )
    else:
        penv = ParallelEnv(
            args.num_workers,
            EnvCreator(lambda: RoboHiveEnv(args.env_name, device=device)),
        )
    if "visual" in args.env_name:
        if args.r3m:
            tenv = TransformedEnv(
                penv,
                R3MTransform(in_keys=["pixels"], download=True, model_name="resnet50"),
            )
        else:
            tenv = penv
    else:
        tenv = penv
    policy = RandomPolicy(tenv.action_spec)  # some random policy

    if torch.cuda.device_count():
        devices = [f"cuda:{i}" for i in range(args.num_collectors)]
    else:
        devices = ["cpu"] * args.num_collectors
    print(devices)
    if args.serial:
        device = devices[0]
        assert args.num_collectors == 1
        collector = SyncDataCollector(
            tenv,
            policy=policy,
            frames_per_batch=args.frames_per_batch,
            total_frames=args.total_frames,
            passing_device=device,
            device=device,
            split_trajs=False,
        )
    else:
        collector = MultiaSyncDataCollector(
            [tenv] * args.num_collectors,
            policy=policy,
            frames_per_batch=args.frames_per_batch,
            total_frames=args.total_frames,
            passing_devices=devices,
            devices=devices,
            split_trajs=False,
        )

    pbar = tqdm.tqdm(total=args.total_frames)
    for i, data in enumerate(collector):
        if i == 3:
            t0 = time.time()
            total = 0
        if i >= 3:
            total += data.numel()
        pbar.update(data.numel())
    t = time.time() - t0
    print(f"{args.env_name}, Time: {t:4.4f}, Rate: {args.total_frames / t: 4.4f} fps")
    del collector
    del tenv
