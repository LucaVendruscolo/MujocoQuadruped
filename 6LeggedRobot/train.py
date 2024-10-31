from BouncyRobot import WalkingRobotEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
import os
import torch
from datetime import datetime
from stable_baselines3 import SAC

# Create the environment
env = WalkingRobotEnv()
device = torch.device("cuda")
print(f"Using device: {device}")

# Get the current timestamp or a custom run name for logging
run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # You can also replace this with a custom string

# Set up the log directory with a unique subdirectory for each run
log_dir = f"tensorboard_logs/{run_id}/"
os.makedirs(log_dir, exist_ok=True)
print(f"Logging to: {log_dir}")

# Create the PPO model with TensorBoard support
#model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device=device)
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

# Configure TensorBoard logger
new_logger = configure(log_dir, ["stdout", "tensorboard"])

# Set the logger to the model
model.set_logger(new_logger)

# Training parameters
timesteps = 100000000
render_frequency = 1  # Render every step
save_interval = 1000000  # Save the model every 1,000,000 steps
obs = env.reset()
cumulative_reward = 0  # Track the cumulative reward for the current episode
total_episodes = 0  # Track the number of episodes

# Start training loop
for step in range(timesteps):
    # Predict action based on the current observation
    action, _states = model.predict(obs)

    # Apply the action and take a step in the environment
    obs, reward, done, info = env.step(action)
    
    # Update cumulative reward for the current episode
    cumulative_reward += reward

    # If the episode is done (either falling or reaching the goal), reset the environment
    if done:
        total_episodes += 1
        print(f"Episode {total_episodes} finished with reward: {cumulative_reward}")

        # Log episode reward and length to TensorBoard
        model.logger.record("episode/length", step)
        model.logger.record("episode/reward", cumulative_reward)
        model.logger.dump(step)

        cumulative_reward = 0  # Reset cumulative reward for the next episode
        obs = env.reset()

    # Save the model every save_interval steps
    if step > 0 and step % save_interval == 0:
        model.save(f"{log_dir}/ppo_walking_robot_step_{step}")
        print(f"Model saved at step {step}")

# Save the final trained model
model.save(f"{log_dir}/ppo_walking_robot_final")

# Close the environment
env.close()
