## Setting up
1. Follow the [installation instructions](https://github.com/facebookresearch/agenthive/blob/dev/GET_STARTED.md) for AgentHive.  
2. Download **FK-v1(expert)** and **DAPG(expert)** dataset from the RoboHive dataset collection - [RoboSet](https://github.com/vikashplus/robohive/wiki/7.-Datasets).

## Behavior Cloning Running Instructions

```
sim_backend=MUJOCO MUJOCO_GL=egl python bc/run_bc_h5.py \
                                    encoder = <visual-encoder> \
                                    cam_name = <camera-name> \
                                    env_name = <env-name> \
                                    from_pixels = True \
                                    data_file = <path-to-dataset>
```
Currently, three visual encoders are supported: [VC1](https://github.com/facebookresearch/eai-vc), [R3M](https://github.com/facebookresearch/r3m), [RRL](https://github.com/facebookresearch/RRL). To use the largest model variant of each one set `encoder=vc1l/r3m50/rrl50`.  
To run the experiments using privileged state or proprioceptive-only information, set `from_pixels=False` and `encoder=state/proprio`, respectively.  

## Results

<img src="https://github.com/facebookresearch/agenthive/blob/dev/scripts/figures/kitchen_human.png" width="450">  <img src="https://github.com/facebookresearch/agenthive/blob/dev/scripts/figures/hms_human.png" width="450">

<img src="https://github.com/facebookresearch/agenthive/blob/dev/scripts/figures/kitchen_expert.png" width="450">  <img src="https://github.com/facebookresearch/agenthive/blob/dev/scripts/figures/robel_expert.png" width="450">

For each of the visual baselines, the results are averaged over 3 camera angles (except for the Robel Suite, where all the camera angles are used to avoid partial observability), 3 seeds, and 25 evaluation trajectories. 

| Benchmark Suite | Dataset Type | Camera Angles | Seeds |
| --- | --- | --- | --- |
| Kitchen (FK-v1) | FK-v1(expert) | `left_cam`,`right_cam`,`top_cam` | `1`,`2`,`3` |
| Kitchen (FK-v1) | FK-v1(human) | `left_cam`,`right_cam`,`top_cam` | `1`,`2`,`3` |
| Hand Manipulation Suite (HMS) | HMS(Human) | `view_1`,`view_4`,`vil_camera` | `1`,`2`,`3` |
| Robel Suite | ROBEL(Expert) | `[A:headCam,A:leftCam,A:tailCam,A:rightCam]` | `1`,`2`,`3` |
