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
from torchrl.collectors.distributed.rpc import RPCDataCollector
from torchrl.collectors.collectors import MultiaSyncDataCollector, RandomPolicy
from torchrl.envs import EnvCreator, ParallelEnv, R3MTransform, TransformedEnv

parser = argparse.ArgumentParser()
parser.add_argument("--num_workers", default=2, type=int)
parser.add_argument("--num_collectors", default=4, type=int)
parser.add_argument("--frames_per_batch", default=200, type=int)
parser.add_argument("--total_frames", default=20_000, type=int)
parser.add_argument("--r3m", action="store_true")
parser.add_argument("--env_name", default="franka_micro_random-v3")

if __name__ == "__main__":
    args = parser.parse_args()

    if args.num_workers > 1:
        penv = ParallelEnv(
            args.num_workers,
            EnvCreator(lambda: RoboHiveEnv(args.env_name, device="cuda:0")),
        )
    else:
        penv = RoboHiveEnv(args.env_name, device="cuda:0")
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

    device = "cuda:0"

    slurm_conf = {
        "timeout_min": 100,
        "slurm_partition": "train",
        "slurm_cpus_per_gpu": 12,
        "slurm_gpus_per_task": 1,
    }

    collector = RPCDataCollector(
        [tenv] * args.num_collectors,
        policy=policy,
        frames_per_batch=args.frames_per_batch,
        total_frames=args.total_frames,
        storing_device=device,
        split_trajs=False,
        sync=True,
        launcher="submitit",
        slurm_kwargs=slurm_conf,
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
