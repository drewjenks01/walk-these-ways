import torch
import time
import random

class CommandProfile:
    def __init__(self, dt, max_time_s=10.):
        self.dt = dt
        self.max_timestep = int(max_time_s / self.dt)
        self.commands = torch.zeros((self.max_timestep, 9))
        self.start_time = 0

    def get_command(self, t):
        timestep = int((t - self.start_time) / self.dt)
        timestep = min(timestep, self.max_timestep - 1)
        return self.commands[timestep, :]

    def get_buttons(self):
        return [0, 0, 0, 0]

    def reset(self, reset_time):
        self.start_time = reset_time


class ConstantAccelerationProfile(CommandProfile):
    def __init__(self, dt, max_speed, accel_time, zero_buf_time=0):
        super().__init__(dt)
        zero_buf_timesteps = int(zero_buf_time / self.dt)
        accel_timesteps = int(accel_time / self.dt)
        self.commands[:zero_buf_timesteps] = 0
        self.commands[zero_buf_timesteps:zero_buf_timesteps + accel_timesteps, 0] = torch.arange(0, max_speed,
                                                                                                 step=max_speed / accel_timesteps)
        self.commands[zero_buf_timesteps + accel_timesteps:, 0] = max_speed


class ElegantForwardProfile(CommandProfile):
    def __init__(self, dt, max_speed, accel_time, duration, deaccel_time, zero_buf_time=0):
        import numpy as np

        zero_buf_timesteps = int(zero_buf_time / dt)
        accel_timesteps = int(accel_time / dt)
        duration_timesteps = int(duration / dt)
        deaccel_timesteps = int(deaccel_time / dt)

        total_time_s = zero_buf_time + accel_time + duration + deaccel_time

        super().__init__(dt, total_time_s)

        x_vel_cmds = [0] * zero_buf_timesteps + [*np.linspace(0, max_speed, accel_timesteps)] + \
                     [max_speed] * duration_timesteps + [*np.linspace(max_speed, 0, deaccel_timesteps)]

        self.commands[:len(x_vel_cmds), 0] = torch.Tensor(x_vel_cmds)


class ElegantYawProfile(CommandProfile):
    def __init__(self, dt, max_speed, zero_buf_time, accel_time, duration, deaccel_time, yaw_rate):
        import numpy as np

        zero_buf_timesteps = int(zero_buf_time / dt)
        accel_timesteps = int(accel_time / dt)
        duration_timesteps = int(duration / dt)
        deaccel_timesteps = int(deaccel_time / dt)

        total_time_s = zero_buf_time + accel_time + duration + deaccel_time

        super().__init__(dt, total_time_s)

        x_vel_cmds = [0] * zero_buf_timesteps + [*np.linspace(0, max_speed, accel_timesteps)] + \
                     [max_speed] * duration_timesteps + [*np.linspace(max_speed, 0, deaccel_timesteps)]

        yaw_vel_cmds = [0] * zero_buf_timesteps + [0] * accel_timesteps + \
                       [yaw_rate] * duration_timesteps + [0] * deaccel_timesteps

        self.commands[:len(x_vel_cmds), 0] = torch.Tensor(x_vel_cmds)
        self.commands[:len(yaw_vel_cmds), 2] = torch.Tensor(yaw_vel_cmds)


class ElegantGaitProfile(CommandProfile):
    def __init__(self, dt, filename):
        import numpy as np
        import json

        with open(f'../command_profiles/{filename}', 'r') as file:
                command_sequence = json.load(file)

        len_command_sequence = len(command_sequence["x_vel_cmd"])
        total_time_s = int(len_command_sequence / dt)

        super().__init__(dt, total_time_s)

        self.commands[:len_command_sequence, 0] = torch.Tensor(command_sequence["x_vel_cmd"])
        self.commands[:len_command_sequence, 2] = torch.Tensor(command_sequence["yaw_vel_cmd"])
        self.commands[:len_command_sequence, 3] = torch.Tensor(command_sequence["height_cmd"])
        self.commands[:len_command_sequence, 4] = torch.Tensor(command_sequence["frequency_cmd"])
        self.commands[:len_command_sequence, 5] = torch.Tensor(command_sequence["offset_cmd"])
        self.commands[:len_command_sequence, 6] = torch.Tensor(command_sequence["phase_cmd"])
        self.commands[:len_command_sequence, 7] = torch.Tensor(command_sequence["bound_cmd"])
        self.commands[:len_command_sequence, 8] = torch.Tensor(command_sequence["duration_cmd"])

class RCControllerProfile(CommandProfile):
    def __init__(self, dt, state_estimator, x_scale=1.0, y_scale=1.0, yaw_scale=1.0, probe_vel_multiplier=1.0):
        super().__init__(dt)
        self.state_estimator = state_estimator
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.yaw_scale = yaw_scale

        self.probe_vel_multiplier = probe_vel_multiplier

        self.triggered_commands = {i: None for i in range(4)}  # command profiles for each action button on the controller
        self.currently_triggered = [0, 0, 0, 0]
        self.button_states = [0, 0, 0, 0]

    def get_command(self, t, probe=False):

        command = self.state_estimator.get_command()
        command[0] = command[0] * self.x_scale
        command[1] = command[1] * self.y_scale
        command[2] = command[2] * self.yaw_scale

        reset_timer = False

        if probe:
            command[0] = command[0] * self.probe_vel_multiplier
            command[2] = command[2] * self.probe_vel_multiplier

        # check for action buttons
        prev_button_states = self.button_states[:]
        self.button_states = self.state_estimator.get_buttons()
        for button in range(4):
            if self.triggered_commands[button] is not None:
                if self.button_states[button] == 1 and prev_button_states[button] == 0:
                    if not self.currently_triggered[button]:
                        # reset the triggered action
                        self.triggered_commands[button].reset(t)
                        # reset the internal timing variable
                        reset_timer = True
                        self.currently_triggered[button] = True
                    else:
                        self.currently_triggered[button] = False
                # execute the triggered action
                if self.currently_triggered[button] and t < self.triggered_commands[button].max_timestep:
                    command = self.triggered_commands[button].get_command(t)


        return command, reset_timer

    def add_triggered_command(self, button_idx, command_profile):
        self.triggered_commands[button_idx] = command_profile

    def get_buttons(self):
        return self.state_estimator.get_buttons()

class RCControllerProfileAccel(RCControllerProfile):
    def __init__(self, dt, state_estimator, x_scale=1.0, y_scale=1.0, yaw_scale=1.0):
        super().__init__(dt, state_estimator, x_scale=x_scale, y_scale=y_scale, yaw_scale=yaw_scale)
        self.x_scale, self.y_scale, self.yaw_scale = self.x_scale / 100., self.y_scale / 100., self.yaw_scale / 100.
        self.velocity_command = torch.zeros(3)

    def get_command(self, t):

        accel_command = self.state_estimator.get_command()
        self.velocity_command[0] = self.velocity_command[0]  + accel_command[0] * self.x_scale
        self.velocity_command[1] = self.velocity_command[1]  + accel_command[1] * self.y_scale
        self.velocity_command[2] = self.velocity_command[2]  + accel_command[2] * self.yaw_scale

        # check for action buttons
        prev_button_states = self.button_states[:]
        self.button_states = self.state_estimator.get_buttons()
        for button in range(4):
            if self.button_states[button] == 1 and self.triggered_commands[button] is not None:
                if prev_button_states[button] == 0:
                    # reset the triggered action
                    self.triggered_commands[button].reset(t)
                # execute the triggered action
                return self.triggered_commands[button].get_command(t)

        return self.velocity_command[:]

    def add_triggered_command(self, button_idx, command_profile):
        self.triggered_commands[button_idx] = command_profile

    def get_buttons(self):
        return self.state_estimator.get_buttons()


class VisionControllerProfile(RCControllerProfile):
    
    def __init__(self, dt, state_estimator, x_scale=1.0, y_scale=1.0, yaw_scale=1.0, random_drift=False):
        super().__init__(dt, state_estimator, x_scale=x_scale, y_scale=y_scale, yaw_scale=2.5)

        # whether or not CommandNet being used
        self.use_commandnet = False

        # which low-level poliocy being used
        self.policy='walk'

        # use yaw in obs or not (for policies)
        self.yaw_bool = False

        # whether XBOX or RC controller used
        self.controller = 0   # 0=RC, 1=XBOX
        self.xbox=None

        # boolean to tell whether camera is working during logging and NN
        # -1 = not initialized, 0 = off, 1 = on
        self.realsense_camera = -1

        # whether using domain randomization
        self.random_drift = random_drift
        if self.random_drift:
            print('Using drift')
            self.domain_change_timer = time.time()



    def get_command(self, t, probe=False):

        commands, reset_timer = super().get_command(t, probe)

        # get status of camera
        self.realsense_camera = self.state_estimator.realsense_camera

        # check if change controller command from rc
            # if mode is to use NN but camera not initialized, then do not change mode
        if self.state_estimator.mode == 7 and self.realsense_camera == -1:
            print('Warning: tried to use NN without camera initialization.')
        else:
            se_mode = self.state_estimator.mode

        #print('received command',self.state_estimator.realsense_commands)

        if self.use_commandnet:
            x_cmd , yaw_cmd, policy = self.state_estimator.realsense_commands
            
            # walk policy
            if policy==0:
                self.policy='walk'
                self.yaw_bool = False
                commands[3] = 0.1
                commands[4] = 3.0
                commands[9] = 0.08

            # stair policy
            elif policy==1:
                self.policy='stairs'
                commands[3] = 0.1
                commands[4] = 2.0       # step freq
                commands[9] = 0.30      # footswing height
                self.yaw_bool = True
            
            # duck policy
            elif policy==2:             
                self.policy='walk'
                self.yaw_bool = False
                commands[3] = -0.2
                commands[4] = 3.0
                commands[9] = 0.08

            # check if commands were being predicted or not, scale and set if so
            if x_cmd != -1.0 and yaw_cmd != -1.0:
                commands[0] = x_cmd
                commands[2] = yaw_cmd

                # multipliers
                commands[0] = commands[0] * self.x_scale
                commands[2] = commands[2] * self.yaw_scale

            if se_mode!=7:
                print('Stopping NN')
                self.use_commandnet=False

        else:

            # override policy if there is a negative x command
            if commands[0]<0:
                se_mode = 0

            if se_mode==4:  # if UP arrow recorded on controller
                self.policy = 'stairs' # stair policy

                self.yaw_bool=True

                commands[3] = 0.1
                commands[4] = 2.0       # step freq
                commands[9] = 0.30      # footswing height


            elif se_mode==5: # if RIGHT arrow record on controller
                self.policy = 'walk' # walk policy

                self.yaw_bool=False
                
                commands[3] = 0.1
                commands[4] = 3.0
                commands[9] = 0.08
            
            
            elif se_mode==6: # if DOWN arrow record on controller
                self.policy = 'walk' # walk policy, duck

                self.yaw_bool=False

                commands[3] = -0.2
                commands[4] = 3.0
                commands[9] = 0.08

            elif se_mode == 7:
                print('Using NN')
                self.use_commandnet=True


        # randomize drifts every 5 seconds
        if self.random_drift and time.time()-self.domain_change_timer>=5.0:
            random_yaw_drift  = self.domain_randomization()

            commands[2] += random_yaw_drift

            self.domain_change_timer = time.time()


        return commands, reset_timer


    def add_triggered_command(self, button_idx, command_profile):
        self.triggered_commands[button_idx] = command_profile

    def get_buttons(self):
        return self.state_estimator.get_buttons()
    
    def domain_randomization(self):
        rand_yaw_drift = random.uniform(0.1,0.3)*random.choice([1,-1])

        print('New drift:',rand_yaw_drift)

        return rand_yaw_drift


class KeyboardProfile(CommandProfile):
    # for control via keyboard inputs to isaac gym visualizer
    def __init__(self, dt, isaac_env, x_scale=1.0, y_scale=1.0, yaw_scale=1.0):
        super().__init__(dt)
        from isaacgym.gymapi import KeyboardInput
        self.gym = isaac_env.gym
        self.viewer = isaac_env.viewer
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.yaw_scale = yaw_scale
        self.gym.subscribe_viewer_keyboard_event(self.viewer, KeyboardInput.KEY_UP, "FORWARD")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, KeyboardInput.KEY_DOWN, "REVERSE")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, KeyboardInput.KEY_LEFT, "LEFT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, KeyboardInput.KEY_RIGHT, "RIGHT")

        self.keyb_command = [0, 0, 0]
        self.command = [0, 0, 0]

    def get_command(self, t):
        events = self.gym.query_viewer_action_events(self.viewer)
        events_dict = {event.action: event.value for event in events}
        print(events_dict)
        if "FORWARD" in events_dict and events_dict["FORWARD"] == 1.0: self.keyb_command[0] = 1.0
        if "FORWARD" in events_dict and events_dict["FORWARD"] == 0.0: self.keyb_command[0] = 0.0
        if "REVERSE" in events_dict and events_dict["REVERSE"] == 1.0: self.keyb_command[0] = -1.0
        if "REVERSE" in events_dict and events_dict["REVERSE"] == 0.0: self.keyb_command[0] = 0.0
        if "LEFT" in events_dict and events_dict["LEFT"] == 1.0: self.keyb_command[1] = 1.0
        if "LEFT" in events_dict and events_dict["LEFT"] == 0.0: self.keyb_command[1] = 0.0
        if "RIGHT" in events_dict and events_dict["RIGHT"] == 1.0: self.keyb_command[1] = -1.0
        if "RIGHT" in events_dict and events_dict["RIGHT"] == 0.0: self.keyb_command[1] = 0.0

        self.command[0] = self.keyb_command[0] * self.x_scale
        self.command[1] = self.keyb_command[2] * self.y_scale
        self.command[2] = self.keyb_command[1] * self.yaw_scale

        print(self.command)

        return self.command


if __name__ == "__main__":
    cmdprof = ConstantAccelerationProfile(dt=0.2, max_speed=4, accel_time=3)
    print(cmdprof.commands)
    print(cmdprof.get_command(2))
