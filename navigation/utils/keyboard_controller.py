
from pynput.keyboard import Key
from pynput import keyboard

class KeyboardController:

    def __init__(self):
        self.command=[0.0,0.0,0.0]
        self.end=False

    def on_press(self,key):
        #print('{0} pressed'.format(
            #key))
        self.check_key(key)

    def on_release(self,key):
        #print('{0} release'.format(
        # key))
        if key == Key.esc:
            # Stop listener
            self.end=True
            return False

    def check_key(self,key):
        if key==Key.up:
            self.command[0]+=1.0
        elif key==Key.down:
            self.command[0]-=1.0
        elif key==Key.left:
            self.command[1]+=1.0
        elif key==Key.right:
            self.command[1]-=1.0
        elif key==Key.esc:
            self.on_release(key)
        elif key.char=='a':
            self.command[2]+=1.0
        elif key.char=='s':
            self.command[2]-=1.0

        print(self.command)
    

    def start_listening(self):
        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        listener.start()