help="
USAGE:
./hydra/train_rs_tasks.sh <task> <target>
    <task> choices:     kitchen, myo, hands, arms
    <target> choices:   '', local, slurm

Example:
./hydra/train_rs_tasks.sh kitchen       # runs natively
./hydra/train_rs_tasks.sh kitchen local # use local launcher
./hydra/train_rs_tasks.sh kitchen slurm # use slurm launcher
"

# Configure launch
if [ "$#" -ne 2 ] ; then
    config=""
else
    config="--multirun hydra/output=$2 hydra/launcher=$2"
fi

# Configure envs
if [ "$1" == "kitchen" ] ; then
    envs="kitchen_micro_open-v3,kitchen_micro_close-v3,kitchen_rdoor_open-v3,kitchen_rdoor_close-v3,kitchen_ldoor_open-v3,kitchen_ldoor_close-v3,kitchen_sdoor_open-v3,kitchen_sdoor_close-v3,kitchen_light_on-v3,kitchen_light_off-v3,kitchen_knob4_on-v3,kitchen_knob4_off-v3,kitchen_knob3_on-v3,kitchen_knob3_off-v3,kitchen_knob2_on-v3,kitchen_knob2_off-v3,kitchen_knob1_on-v3,kitchen_knob1_off-v3,franka_micro_open-v3,franka_micro_close-v3,franka_micro_random-v3,franka_slide_open-v3,franka_slide_close-v3,franka_slide_random-v3,kitchen_micro_slide_close-v3,kitchen_openall-v3,kitchen_closeall-v3"

    config="--config-name hydra_kitchen_config.yaml $config"

elif [ "$1" == "kitchen-v4" ] ; then
    envs="FK1_RelaxRandom-v4,FK1_MicroOpenRandom-v4,FK1_MicroCloseRandom-v4,FK1_RdoorOpenRandom-v4,FK1_RdoorCloseRandom-v4,FK1_LdoorOpenRandom-v4,FK1_LdoorCloseRandom-v4,FK1_SdoorOpenRandom-v4,FK1_SdoorCloseRandom-v4,FK1_LightOnRandom-v4,FK1_LightOffRandom-v4,FK1_Knob4OnRandom-v4,FK1_Knob4OffRandom-v4,FK1_Knob3OnRandom-v4,FK1_Knob3OffRandom-v4,FK1_Knob2OnRandom-v4,FK1_Knob2OffRandom-v4,FK1_Knob1OnRandom-v4,FK1_Knob1OffRandom-v4,FK1_Stove1KettleRandom-v4,FK1_Stove4KettleRandom-v4"

    config="--config-name hydra_kitchen_config.yaml $config"


elif [ "$1" == "myo" ] ; then
    envs="motorFingerReachFixed-v0,motorFingerReachRandom-v0,myoFingerReachFixed-v0,myoFingerReachRandom-v0,myoHandReachFixed-v0,myoHandReachRandom-v0,motorFingerPoseFixed-v0,motorFingerPoseRandom-v0,myoFingerPoseFixed-v0,myoFingerPoseRandom-v0,myoElbowPose1D6MFixed-v0,myoElbowPose1D6MRandom-v0,myoElbowPose1D6MExoRandom-v0,myoElbowPose1D6MExoRandom-v0,myoHandPoseFixed-v0,myoHandPoseRandom-v0,myoHandKeyTurnFixed-v0,myoHandKeyTurnRandom-v0,myoHandObjHoldFixed-v0,myoHandObjHoldRandom-v0,myoHandPenTwirlFixed-v0,myoHandPenTwirlRandom-v0,myoHandBaodingFixed-v1,myoHandBaodingRandom-v1,myoHandBaodingFixed4th-v1,myoHandBaodingFixed8th-v1,myoHandDieReorientFixed-v0,myoHandDieReorientRandom-v0"
    envs+=",myoChallengeDieReorientDemo-v0,myoChallengeDieReorientP1-v0,myoChallengeDieReorientP2-v0,myoChallengeBaodingP1-v1,myoChallengeBaodingP2-v1"

    config="--config-name hydra_myo_config.yaml $config"

elif [ "$1" == "hands" ] ; then
    envs="pen-v1,door-v1,hammer-v1,relocate-v1,baoding-v1,baoding4th-v1,baoding8th-v1"

    # envs+=",pen-v0,door-v0,hammer-v0,relocate-v0,baoding-v0"

    # envs="baodingH200-v1,baoding4thH200-v1,baoding8thH200-v1,baodingH100-v1,baoding4thH100-v1,baoding8thH100-v1,baodingH60-v1,baoding4thH60-v1,baoding8thH60-v1"

    config="--config-name hydra_hand_config.yaml $config"

elif [ "$1" == "arms" ] ; then
    envs="FrankaReachFixed-v0,FrankaReachRandom-v0,FrankaPushFixed-v0,FrankaPushRandom-v0,FetchReachFixed-v0,FetchReachRandom-v0"
    config="--config-name hydra_arms_config.yaml $config"

elif [ "$1" == "quads" ] ; then
    envs="DKittyWalkFixed-v0, DKittyWalkRandom-v0"
    config="--config-name hydra_quads_config.yaml $config"


else
    # echo $help
    printf "$help"
    exit 0
fi

# Disp NPG commands
echo "NPG: ======="
echo "python hydra_mjrl_launcher.py --config-path config $config env=$envs"

# Disp SAC commands
echo "SAC: ======="
echo "python launcher_vrl.py --config-path config $config env=$envs env_args.from_pixels=False algos=sac"