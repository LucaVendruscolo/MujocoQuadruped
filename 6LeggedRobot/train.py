from BouncyRobot import WalkingRobotEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
import os
import torch
from datetime import datetime
from stable_baselines3 import SAC

env = WalkingRobotEnv()
device = torch.device("cuda")
print(f"Using device: {device}")

run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 


log_dir = f"tensorboard_logs/{run_id}/"
os.makedirs(log_dir, exist_ok=True)
print(f"Logging to: {log_dir}")


#model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device=device)
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

new_logger = configure(log_dir, ["stdout", "tensorboard"])

model.set_logger(new_logger)

timesteps = 100000000
render_frequency = 1  
save_interval = 1000000 
obs = env.reset()
cumulative_reward = 0  
total_episodes = 0 

for step in range(timesteps):
    action, _states = model.predict(obs)

    obs, reward, done, info = env.step(action)
    
    cumulative_reward += reward

    if done:
        total_episodes += 1
        print(f"Episode {total_episodes} finished with reward: {cumulative_reward}")

        model.logger.record("episode/length", step)
        model.logger.record("episode/reward", cumulative_reward)
        model.logger.dump(step)

        cumulative_reward = 0  


    if step > 0 and step % save_interval == 0:
        model.save(f"{log_dir}/ppo_walking_robot_step_{step}")
        print(f"Model saved at step {step}")

model.save(f"{log_dir}/ppo_walking_robot_final")

env.close()
