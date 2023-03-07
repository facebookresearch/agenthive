# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

os.environ["sim_backend"] = "MUJOCO"

import argparse
import time

import tqdm

from rlhive.rl_envs import RoboHiveEnv
from torchrl.collectors.collectors import MultiaSyncDataCollector, RandomPolicy
from torchrl.envs import EnvCreator, ParallelEnv, R3MTransform, TransformedEnv

parser = argparse.ArgumentParser()
parser.add_argument("--num_workers", default=2, type=int)
parser.add_argument("--num_collectors", default=4, type=int)
parser.add_argument("--frames_per_batch", default=50, type=int)
parser.add_argument("--total_frames", default=20_000, type=int)
parser.add_argument("--r3m", action="store_true")
parser.add_argument("--env_name", default="franka_micro_random-v3")

if __name__ == "__main__":
    args = parser.parse_args()

    penv = ParallelEnv(
        args.num_workers,
        EnvCreator(lambda: RoboHiveEnv(args.env_name, device="cuda:0")),
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
        # tenv.transform[-1].init_stats(reduce_dim=(0, 1), cat_dim=1,
        #                               num_iter=1000)
    policy = RandomPolicy(tenv.action_spec)  # some random policy

    devices = [f"cuda:{i}" for i in range(args.num_collectors)]
    print(devices)
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
