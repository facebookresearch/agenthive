# Performance of NPG on Kitchen tasks for V0.5 release
- In [V0.1](https://github.com/vikashplus/mjrl_dev/blob/redesign/mjrl_dev/agents/v0.1/kitchen/NPG/FinalPerf-NPG.pdf) we notice that `random_init` performs on par with `fixed_init`. Moving forward we are defaulting to `random_init`.
- Results for v0.2 are provided for `RANDOM_ENTRY_POINT`.
Note: Inference with respect to relay datasets will probably still need `fixed_init` to initialize in the demo distribution.

## Results
V0.5 (3 seeds + Comparision wrt V0.4)
- [Final performance](FinalPerf-NPG.pdf)
- [Training curves](TrainPerf-NPG.pdf)

## Known Issues
- N/A

## Hashes
```
83d35df95eb64274c5e93bb32a0a4e2f6576638a (mjrl)
4b76549ad07a57638724bb75e2fb2fd939f60085 (robohive)
2ef4b752e85782f84fa666fce10de5231cc5c917 robohive/simhive/Adroit (v0.1-2-g2ef4b75)
46edd9c361061c5d81a82f2511d4fbf76fead569 robohive/simhive/YCB_sim (heads/main)
b8531308fa34d2bd637d9df468455ae36e2ebcd3 robohive/simhive/dmanus_sim (heads/correct_bracket)
58d561fa416b6a151761ced18f2dc8f067188909 robohive/simhive/fetch_sim (heads/master)
82aaf3bebfa29e00133a6eebc7684e793c668fc1 robohive/simhive/franka_sim (v0.1-7-g82aaf3b)
afb802750acb97f253b2f6fc6e915627d04fcf67 robohive/simhive/furniture_sim (v0.1-20-gafb8027)
cde1442b92523feb26786a38b0b11ba2a9429dd3 robohive/simhive/myo_sim (heads/main)
87cd8dd5a11518b94fca16bc22bb04f6836c6aa7 robohive/simhive/object_sim (87cd8dd)
68030f77f73e247518f9620ab0eed01286ace7b4 robohive/simhive/robel_sim (heads/experimental-hand)
854d0bfb4e48b076e1d2aa4566c2e23bba17ebae robohive/simhive/robotiq_sim (heads/main)
affaf56d56be307538e5eed34f647586281762b2 robohive/simhive/sawyer_sim (heads/master)
f73acd52f3546939828d750744704603de03edf3 sims/scene_sim (heads/master)
49e689ee8d18f5e506ba995aac99822b66700b2b robohive/simhive/trifinger_sim (heads/main)
```