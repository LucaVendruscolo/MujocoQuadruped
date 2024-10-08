import mujoco
import glfw
from gym import Env, spaces
import numpy as np
import time
import math

class WalkingRobotEnv(Env):
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path('FirstRobot\\robot_converted.xml')
        self.data = mujoco.MjData(self.model)

        if not glfw.init():
            raise Exception("Failed to initialize GLFW")
        
        self.Togglerender = False

        if self.Togglerender:
            self.window = glfw.create_window(1200, 900, "MuJoCo Walking Robot", None, None)
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)  

            self.camera = mujoco.MjvCamera()
            self.option = mujoco.MjvOption()

            self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
            self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

            mujoco.mjv_defaultCamera(self.camera)
            mujoco.mjv_defaultOption(self.option)

            
            self.camera.lookat[:] = [0, 0, 0]  
            self.camera.distance = 1.0  
            self.camera.elevation = -20  
            self.camera.azimuth = 45 

        self.target_fps = 20000
        self.time_per_frame = 1.0 / self.target_fps

      
        obs_size = self.model.nq + self.model.nv  
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        self.totReward = 0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)

        self.torque_limit = 1.0  
        self.maxForwardDisplacement = 0

        self.total_steps = 0
        self.total_substeps = 0
        self.start_time = time.perf_counter()

        self.is_raising = True  # Start by rewarding upward motion
        self.last_body_z_pos = 0

        self.steps_in_current_cycle = 0  # To count steps in the current cycle
        self.total_cycles = 0  # To count how many cycles have been completed
        self.total_steps_in_all_cycles = 0 

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = self.torque_limit * action

        substeps = 200  # Number of substeps
        target_time_per_frame = 1.0 / self.target_fps  # Time per frame based on target FPS

        for _ in range(substeps):
            # Start timing for each substep (treated as a frame)
            start_time = time.perf_counter()

            mujoco.mj_step(self.model, self.data)  # Step through simulation
            self.stepCount += 1  # Each substep counts as a step/frame
            

            if self.Togglerender:
                self.render()

                # End timing for each substep
                end_time = time.perf_counter()

                # Calculate elapsed time for the substep
                elapsed_time = end_time - start_time

                # If the elapsed time is less than the target frame time, sleep for the remaining time
                if elapsed_time < target_time_per_frame:
                    time.sleep(target_time_per_frame - elapsed_time)

                # Calculate and print the actions per second after each substep
                actions_per_second = 1 / (time.perf_counter() - start_time)
                #print(f"Actions per second: {actions_per_second:.2f}",  self.data.qpos[2])

        # Gather observation and calculate reward after all substeps
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        reward = self.compute_reward()
        done = self.is_done()
        self.totReward += reward
        #print("self.totReward" , self.totReward)

        return obs, reward, done, {}
    

    def compute_reward(self):

        target_top = 0.15
        target_bottom = 0.05
        body_z_pos = self.data.qpos[2]


        if self.is_raising:
            reward = (self.last_body_z_pos - body_z_pos) * -20
    
        else:
            reward = (self.last_body_z_pos - body_z_pos) * 20

        if self.is_raising and body_z_pos >= target_top:
            reward += 20
            self.is_raising = False  
            self.min_body_z_pos = body_z_pos
            

        elif not self.is_raising and body_z_pos <= target_bottom:
            reward += 20
            self.is_raising = True 
            self.max_body_z_pos = body_z_pos  


        torque_penalty = np.sum(np.square(self.data.ctrl)) 
        reward -= 0.001 * torque_penalty 

        self.last_body_z_pos = self.data.qpos[2]
        self.steps_in_current_cycle += 1     

        return reward




    def is_done(self):
        forward_distance = self.data.qpos[0]
        
        # If the episode is finished (reaching a distance or step limit)
        if self.stepCount > 500000 or forward_distance >= 10.0 or forward_distance <= -10.0:
            self.stepCount = 0 
            self.maxForwardDisplacement = 0
            self.totReward = 0

            # Calculate average steps per cycle
            if self.total_cycles > 0:
                avg_steps_per_cycle = self.total_steps_in_all_cycles / self.total_cycles
                print(f"Average steps per up-down cycle: {avg_steps_per_cycle:.2f}")

            # Reset counters for the next episode
            self.total_cycles = 0
            self.total_steps_in_all_cycles = 0
            self.steps_in_current_cycle = 0

            return True

        return False


    def render(self):

        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)

        
      
        if viewport_width > 0 and viewport_height > 0:
            viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

            
            mujoco.mjv_updateScene(self.model, self.data, self.option, None, self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)

            mujoco.mjr_render(viewport, self.scene, self.context)

 
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        else:
            print("Viewport dimensions are zero. Check if the window is created properly.")




    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.maxForwardDisplacement = 0
        self.stepCount = 0
        self.is_raising = True  # Reset to raising mode
        self.max_body_z_pos = self.data.qpos[2]  # Initialize to the current position
        self.min_body_z_pos = self.data.qpos[2]  # Initialize to the current position
        return np.concatenate([self.data.qpos, self.data.qvel])
    


    def close(self):
        
        glfw.terminate()
        mujoco.mjr_freeContext(self.context)
        mujoco.mjv_freeScene(self.scene)
