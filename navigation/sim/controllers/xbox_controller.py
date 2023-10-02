
class XboxController(object):
    import math
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self):

        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.YDPad=0
        self.UPad = 0
        self.DPad = 0
        self.XDPad = 0
        self.LDPad = 0
        self.RDPad = 0
        self.thumbs=0

        import threading
        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()


    def read(self): # return the buttons/triggers that you care about in this methode
        controls = {}

        # joysticks
        controls['y_vel'] = self.LeftJoystickY # y vel
        controls['yaw'] = self.RightJoystickX # yaw vel

        # buttons
        controls['y_but'] = self.Y
        controls['x_but'] = self.X

        # triggers
        controls['l_trig']=self.LeftTrigger
        controls['y_vr_trigel']=self.RightTrigger

        # D pad
        controls['r_dpad']=self.RDPad
        controls['l_dpad']=self.LDPad
        controls['up_dpad']=self.UDPad
        controls['down_dpad']=self.DDPad
        

        return controls


    def _monitor_controller(self):
        from inputs import get_gamepad
        while True:
            events = get_gamepad()
            for event in events:
                
                # IN USE

                # left joystick up/down: controls robot y_vel
                if event.code == 'ABS_Y':
                    self.LeftJoystickY = -1*event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                
                # right joystick left/right: controls robot yaw_vel
                elif event.code == 'ABS_RX':
                    self.RightJoystickX = -1*event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
               
                # left trigger: stops using a NN action policy if on
                elif event.code == 'ABS_Z':
                    self.LeftTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1

                # right trigger: starts using a NN action policy
                elif event.code == 'ABS_RZ':
                    self.RightTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1

                # X button: hard resets the current demo and environment
                elif event.code == 'BTN_NORTH':
                    self.X = event.state #previously switched with X
                
                # Y button: starts recording a demo
                elif event.code == 'BTN_WEST':
                    self.Y = event.state #previously switched with Y

                # D Pad left/right: right = walk gait, left = nothing
                elif event.code == 'ABS_HAT0X':
                    self.XDPad = event.state / XboxController.MAX_JOY_VAL
                    if self.XDPad<0:
                        self.LDPad = 1
                        self.RDPad = 0
                    elif self.XDPad>0:
                        self.LDPad = 0
                        self.RDPad = 1
                
                # D Pad up/down: up = climb gait, down = duck gait
                elif event.code == 'ABS_HAT0Y':
                    self.YDPad = event.state / XboxController.MAX_JOY_VAL *-1
                    if self.YDPad<0:
                        self.DDPad = 1
                        self.UDPad = 0
                    elif self.YDPad>0:
                        self.DDPad = 0
                        self.UDPad = 1


                ##########################
                
                # NOT IN USE
                elif event.code == 'ABS_X':
                    self.LeftJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RY':
                    self.RightJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'BTN_TL':
                    self.LeftBumper = event.state
                elif event.code == 'BTN_TR':
                    self.RightBumper = event.state 
                elif event.code == 'BTN_EAST':
                    self.B = event.state
                elif event.code == 'BTN_THUMBL':
                    self.thumbs = event.state
                elif event.code == 'BTN_THUMBR':
                    self.thumbs = event.state*-1
                elif event.code == 'BTN_SELECT':
                    self.Back = event.state
                elif event.code == 'BTN_START':
                    self.Start = event.state
                elif event.code == 'BTN_SOUTH':
                    self.A = event.state
