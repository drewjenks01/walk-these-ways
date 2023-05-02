
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
        self.XDPad = 0
        self.DownDPad = 0
        self.thumbs=0

        import threading
        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()


    def read(self): # return the buttons/triggers that you care about in this methode
        x = -1 * self.LeftJoystickX  # x vel
        y = -1*self.LeftJoystickY # y vel
        yaw = -2*self.RightJoystickX # yaw vel
        a = self.A # switch gaits
        lb = self.LeftBumper
        rb = self.RightBumper
        y_cmd = self.Y
        x_cmd= self.X
        ltrig=self.LeftTrigger
        rtrig=self.RightTrigger
        b=self.B
        thumbs=self.thumbs
        return [y, x, yaw, a,lb,rb,y_cmd, x_cmd,ltrig,rtrig,b,thumbs, self.XDPad, self.YDPad]


    def _monitor_controller(self):
        from inputs import get_gamepad
        while True:
            events = get_gamepad()
            for event in events:
                #print(event.code)
                # up/down
                if event.code == 'ABS_Y':
                    self.LeftJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1

                # left/right
                elif event.code == 'ABS_X':
                    self.LeftJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RY':
                    self.RightJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                
                # yaw left/right
                elif event.code == 'ABS_RX':
                    self.RightJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
               
                # record action script
                elif event.code == 'ABS_Z':
                    self.LeftTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1

                # execute action script
                elif event.code == 'ABS_RZ':
                    self.RightTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1

                # decrease foot swing
                elif event.code == 'BTN_TL':
                    self.LeftBumper = event.state
                
                # increase foot swing
                elif event.code == 'BTN_TR':
                    self.RightBumper = event.state 

                # starts test collection
                elif event.code == 'BTN_SOUTH':
                    self.A = event.state

                # terminates demo
                elif event.code == 'BTN_NORTH':
                    self.X = event.state #previously switched with X
                
                # starts data collection
                elif event.code == 'BTN_WEST':
                    self.Y = event.state #previously switched with Y
                
                # starts video recording
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

                elif event.code == 'ABS_HAT0X':
                    self.XDPad = event.state / XboxController.MAX_JOY_VAL

                elif event.code == 'ABS_HAT0X':
                    self.XDPad = event.state / XboxController.MAX_JOY_VAL
                elif event.code == 'ABS_HAT0Y':
                    self.YDPad = event.state / XboxController.MAX_JOY_VAL *-1
                elif event.code == 'ABS_HAT0Y':
                    self.YDPad = event.state / XboxController.MAX_JOY_VAL *-1
