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

        # self.window = glfw.create_window(1200, 900, "MuJoCo Walking Robot", None, None)
        # glfw.make_context_current(self.window)
        # glfw.swap_interval(1)  

        # self.camera = mujoco.MjvCamera()
        # self.option = mujoco.MjvOption()

        # self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
        # self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        # mujoco.mjv_defaultCamera(self.camera)
        # mujoco.mjv_defaultOption(self.option)

        
        # self.camera.lookat[:] = [0, 0, 0]  
        # self.camera.distance = 1.0  
        # self.camera.elevation = -20  
        # self.camera.azimuth = 45 

        self.target_fps = 60000
        self.time_per_frame = 1.0 / self.target_fps

      
        obs_size = self.model.nq + self.model.nv  
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

       
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)

        self.torque_limit = 1.0  
        self.maxForwardDisplacement = 0

        self.total_steps = 0
        self.total_substeps = 0
        self.start_time = time.perf_counter()

    def step(self, action):
        # Start timing
        start_time = time.perf_counter() 

        action = np.clip(action, self.action_space.low, self.action_space.high)    
        self.data.ctrl[:] = self.torque_limit * action

        substeps = 100  # Increase this value to run more substeps
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)
            #self.render()
            
        self.stepCount += 1
        mujoco.mj_step(self.model, self.data)       
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        reward = self.compute_reward()
        done = self.is_done()

        # End timing
        # end_time = time.perf_counter()

        # Calculate elapsed time and actions per second
        # elapsed_time = end_time - start_time
        # actions_per_second = 1 / elapsed_time  # Since each `step` corresponds to one action
        # print(f"Actions per second: {actions_per_second:.2f}")

        return obs, reward, done, {}


    def compute_reward(self):
        print1 = False
        forward_displacement = self.data.qpos[0]  
        if self.maxForwardDisplacement < forward_displacement:
            reward = (forward_displacement- self.maxForwardDisplacement) * 1000
            self.maxForwardDisplacement = forward_displacement
            print1 = True

        else:
            reward = 0

       
        torque_penalty = np.sum(np.square(self.data.ctrl))  
        reward -= 0.01 * torque_penalty  
        
        

        body_z_pos = self.data.qpos[2]  
        target_height = 0.2 
 
       
        height_reward = -np.abs(body_z_pos - target_height)*0.1  
        reward += height_reward  

        # if print1:
        #     print("reward", reward)
        #     print("torque reward" , -0.01 * torque_penalty )
        #     print("height reward" , height_reward)
        # Penalty for falling

        return reward


    def is_done(self):

          
        forward_distance = self.data.qpos[0]
        if self.stepCount > 10000 or forward_distance >= 10.0 or forward_distance <= -10.0:
            self.stepCount = 0 
            self.maxForwardDisplacement =0
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
        
  
        self.cumulative_reward = 0
        self.stepCount = 0
        
        return np.concatenate([self.data.qpos, self.data.qvel])
    


    def close(self):
        
        glfw.terminate()
        mujoco.mjr_freeContext(self.context)
        mujoco.mjv_freeScene(self.scene)
