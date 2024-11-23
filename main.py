
from core.world.env_factory import EnvFactory
import core.world.sim.drive_env as base_env
from core.evaluate_module import Eval_module

from akida import load_model

def evaluate_akida():

    model_akida = load_model("../akida_models/akida_model.fbz")
    model_akida.summary()


def evaluate_cpu():
    env_name = base_env
    track_type = "boxes"
    track_name = "big_S"

    env_factory = EnvFactory(env_name)

    env = env_factory.createEnv(track_type, track_name, None)

    actor_state = "./ppo_actor.pth"

    evaluator = Eval_module(env, actor_state, "ANN")

    mean_reward = evaluator.eval_policy_ANN(n_eval_episodes=5)

    print("mean_reward :", mean_reward)

def main():

    model = None

    evaluate_akida()

    #evaluate_cpu()

if __name__ == "__main__":

    main()