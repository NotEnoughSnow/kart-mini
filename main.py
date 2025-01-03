
from core.env_factory import EnvFactory
import core.sim.steer_env as base_env
from core.evaluate_module import Eval_module

from akida_models.model_io import load_model
import numpy as np

from akida import devices

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
        0.99765169, 0.06849162,
    ]], dtype=np.float32)

    # ...

    logits = model_akida.predict(obs)

    print(logits)


def verify_akida():

    model_akida = load_model("../akida_models/new.fbz")

    device = devices()[0]
    print("device :", device)
    print("version :", device.version)
    print("ip version: :", device.version)

    print("model ip version: ", model_akida.ip_version)

    # Assuming model_akida is already loaded
    model_akida.map(device)

    # Optionally, print mapping details
    print(f"Model mapped to device: {device}")

    model_akida.summary()


def evaluate_cpu(env):

    actor_state = "./ppo_actor.pth"

    evaluator = Eval_module(env, actor_state, "ANN")

    mean_reward = evaluator.eval_policy_ANN(n_eval_episodes=5)

    print("mean_reward :", mean_reward)

def main():

    env_name = base_env
    track_type = "loader"
    track_name = "big_S"

    env_factory = EnvFactory(env_name)

    env = env_factory.createEnv(track_type, track_name, None)

    #evaluate_akida(env=env)

    verify_akida()

    #evaluate_cpu(env=env)

if __name__ == "__main__":

    main()