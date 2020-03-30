import gym
import os, datetime, argparse
import numpy as np
import pytz
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy


log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)
# parameters for callback
num_update = 0
best_mean_reward = -np.inf

def args():
    """
    Return all training arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="CartPole-v1", help='environment ID')
    return parser.parse_args()

def print_message(message, file_name="out.txt"):
    """print current log on console; also save the log to the disk

    Parameters:
        message(str) -- log message
        file_name(str) -- specify the name of log file
    """
    print(message)  # print the message
    with open(os.path.join(log_dir, file_name), "a") as log_file:
      log_file.write('%s\n' % message)  # save the message

def callback(_locals, _globals):
    """
    callback function for monitoring training, which is called at given stages of the training procedure. It allows to do auto saving, model manipulation, progress bars, and so forth.

    Parameters:
        _locals(dict) -- local variables in the model
        _global(dict) -- global variables in the model

    Return:
        (bool) -- If your callback returns False, training is aborted early.
    """
    global num_update, best_mean_reward
    # process every 1000 steps
    if (num_update + 1) % 1000 == 0:
        # get the result array
        _, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(y) > 0:
            # save the model if best reward is gained
            mean_reward = np.mean(y[-100:])
            update_model = mean_reward > best_mean_reward
            if update_model:
                best_mean_reward = mean_reward
                _locals['self'].save('best_model')

            # output logs
            m = "time: {}, num_update: {}, mean: {:.2f}, best_mean: {:.2f}, model_update: {}".format(
                datetime.datetime.now(pytz.timezone('Asia/Tokyo')),
                num_update, mean_reward, best_mean_reward, update_model)
            print_message(m)
    num_update += 1
    return True

def main(args):
    # Create and wrap the environment
    env = gym.make(args.env)
    env = DummyVecEnv([lambda: env]) # alter to a vector environment, change if running in multi-processes

    # define the model
    model = PPO2('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
    #model = PPO2.load("pretrained_model")

    # train
    model.learn(total_timesteps=100000, callback=callback)

    # test
    state = env.reset()
    for i in range(200):
        # display env
        env.render()

        # inference
        action, _ = model.predict(state)

        # take an action
        state, rewards, done, info = env.step(action)

        # episode finished
        if done:
            break

    # close the environment
    env.close()

if __name__ == "__main__":
  args = args()
  main(args)
    