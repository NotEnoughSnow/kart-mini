
from core.world.env_factory import EnvFactory
import core.world.sim.drive_env as base_env
from core.evaluate_module import Eval_module

from akida_models.model_io import load_model
import numpy as np

import core.snn_utils as SNN_utils

import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def evaluate_akida(env):

    model_akida = load_model("../akida_models/SMRE_random.fbz")
    model_akida.summary()

    obs = np.array([[
        1.16666667, 1.16666667, 1.16666667, 1.16666667, 1.16666667, 1.16666667,
        1.16666667, 1.16666667, 1.16666667, 1.16666667, 1.16666667, 1.16666667,
        1.16666667, 1.16666667, 1.16666667, 1.16666667, 1.16666667, 1.16666667,
        1.16666667, 1.16666667, 1.16666667, 1.16666667, 1.16666667, 1.16666667,
        1.16666667, 1.16666667, 1.16666667, 1.16666667, 1.16666667, 1.16666667,
        1.16666667, 1.16666667, 1.16666667, 1.16666667, 1.16666667, 1.16666667,
        1.16666667, 1.16666667, 1.16666667, 1.16666667, 1.16666667, 1.16666667,
        1.16666667, 1.16666667, 1.16666667, 1.16666667, 1.16666667, 1.16666667,
        1.16666667, 1.16666667, 1.16666667, 1.16666667, 1.16666667, 1.16666667,
        1.16666667, 1.16666667, 1.16666667, 1.16666667, 1.16666667, 0.34192791,
        1.07690352, 0.75227227, 0.31401242, 0.94941888, -3.75118563, 0.77804336,
        0.99765169, 0.06849162
    ]], dtype=np.float32)

    threshold = torch.tensor(env.high)
    shift = np.abs(env.low)

    obs_st = SNN_utils.generate_spike_trains(obs,
                                             num_steps=50,
                                             threshold=threshold,
                                             shift=shift)

    obs_st_int = obs_st.astype(np.int8)

    obs_reshaped = obs_st_int.reshape(50, 1, 1, 68)

    print(obs_reshaped.shape)

    logits = model_akida.predict(obs_reshaped)

    print(logits)


def verify_akida():

    model_akida = load_model("../akida_models/SMRE_random.fbz")
    model_akida.summary()

    # Assuming model_akida is already loaded
    device_name = "PCIe/NSoC_v2/0"  # This is the detected Akida device
    model_akida.map(device_name=device_name)

    # Optionally, print mapping details
    print(f"Model mapped to device: {device_name}")




def evaluate_cpu(env):

    actor_state = "./ppo_actor.pth"

    evaluator = Eval_module(env, actor_state, "ANN")

    mean_reward = evaluator.eval_policy_ANN(n_eval_episodes=5)

    print("mean_reward :", mean_reward)

def main():

    env_name = base_env
    track_type = "boxes"
    track_name = "big_S"

    env_factory = EnvFactory(env_name)

    env = env_factory.createEnv(track_type, track_name, None)

    #evaluate_akida(env=env)

    verify_akida()

    #evaluate_cpu(env=env)

if __name__ == "__main__":

    main()