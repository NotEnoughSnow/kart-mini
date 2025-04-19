
from core.env_factory import EnvFactory
import core.sim.steer_env as base_env
from core.evaluate_module import Eval_module
from torch.distributions import MultivariateNormal, Categorical

from akida_models.model_io import load_model
import numpy as np

from akida import devices

from core.arguments import get_args


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def evaluate_akida(env, n_eval_episodes):

    model_akida = load_model("../akida_models/SMRE_random.fbz")
    model_akida.summary()

    rewards = []

    for episode in range(n_eval_episodes):
        obs, _ = env.reset(options={})
        terminated = False
        truncated = False
        episode_reward = 0


        while not (terminated or truncated):
            # Get action from the actor (policy network)

            int_obs = obs.astype(np.int16)

            logits = model_akida.predict(int_obs)

            dist = Categorical(logits=logits)
            action = dist.sample().detach().numpy()


            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)
        print("reward for this episode :", episode_reward)

    mean_reward = np.mean(rewards)
    return mean_reward


def verify_akida():

    model_akida = load_model("../akida_models/new.fbz")

    device = devices()[0]
    print("device :", device)
    print("version :", device.version)
    print("ip version: :", device.version)

    print("model ip version: ", model_akida.ip_version)

    model_akida.map(device)

    print(f"Model mapped to device: {device}")

    model_akida.summary()


def evaluate_cpu(env):

    actor_state = "./ppo_actor.pth"

    evaluator = Eval_module(env, actor_state, "ANN")

    mean_reward = evaluator.eval_policy_ANN(n_eval_episodes=5)

    print("mean_reward :", mean_reward)

def main(args):

    env_name = base_env
    track_type = "loader"
    track_name = "big_S"

    env_factory = EnvFactory(env_name)

    env = env_factory.createEnv(track_type, track_name, None)

    if args.mode == "verify":
        print("Verifying akida model")
        verify_akida()
    if args.mode == "akida":
        print("Evaluating on akida")
        evaluate_akida(env=env, n_eval_episodes=5)
    if args.mode == "cpu":
        print("Evaluating on cpu")
        evaluate_cpu(env=env)

if __name__ == "__main__":
    args = get_args()
    main(args)