from ale_python_interface import ALEInterface
import numpy as np
import time
import random

try:
    from Tkinter import *
    import ImageTk
    from PIL import Image

except ImportError:
    print("Machine does not have libraries for rendering")

class AtariEnvironment:

    def __init__(self, rom):
        self.ale = ALEInterface()
        self.ale.setFloat(b'repeat_action_probability', 0.0)
        self.ale.loadROM(rom_file=rom)
        self.action_space = self.ale.getMinimalActionSet()
        self.obs = self.reset()

        try:
            self.im = Image.fromarray(self.obs)
            self.root = Tk()
            self.tkim = ImageTk.PhotoImage(self.im)
            self.window = Label(image=self.tkim)
            self.window.image = self.tkim
            self.window.pack()

        except AttributeError:
            print("Cannot create rendering attributes")


    def step(self, action):
        reward = 0.

        # Use if you want environment to provide every 4th frame and repeat action in between
        for i in range(4):
            reward += float(self.ale.act(self.action_space[action]))
            if i == 2:
                frame1 = self.ale.getScreenGrayscale()
            if i == 3:
                frame2 = self.ale.getScreenGrayscale()

        self.obs = np.squeeze(np.maximum(frame1, frame2))

        # Use if you want to receive every frame from environment
        # reward += float(self.ale.act(self.action_space[action]))
        # self.obs = np.squeeze(self.ale.getScreenGrayscale())

        done = self.ale.game_over()
        return self.obs, reward, done

    def reset(self):
        self.ale.reset_game()
        self.obs = np.squeeze(self.ale.getScreenGrayscale())
        return self.obs

    def render(self, rate=0.1):
        self.im = Image.fromarray(self.obs)
        self.tkim = ImageTk.PhotoImage(self.im)
        self.window.configure(image=self.tkim)
        self.window.image = self.tkim
        self.window.update_idletasks()
        self.window.update()
        time.sleep(rate)

    def sample_action(self):
        action = random.choice([0, 1, 2, 3])
        return action


# An observation in this version consists of only the last frame.
# So, velocity cannot be inferred from a single observation.
class POAtariEnvironment():

    def __init__(self, rom):
        self.ale = ALEInterface()
        self.ale.setFloat(b'repeat_action_probability', 0.0)
        self.ale.loadROM(rom_file=rom)
        self.action_space = self.ale.getMinimalActionSet()
        self.obs = self.reset()

        try:
            self.im = Image.fromarray(self.obs)
            self.root = Tk()
            self.tkim = ImageTk.PhotoImage(self.im)
            self.window = Label(image=self.tkim)
            self.window.image = self.tkim
            self.window.pack()

        except AttributeError:
            print("Cannot create rendering attributes")


    def step(self, action):
        reward = 0.

        # Use if you want environment to provide every 4th frame and repeat action in between
        for i in range(4):
            reward += float(self.ale.act(self.action_space[action]))

        frame = self.ale.getScreenGrayscale()

        self.obs = np.squeeze(frame)

        # Use if you want to receive every frame from environment
        # reward += float(self.ale.act(self.action_space[action]))
        # self.obs = np.squeeze(self.ale.getScreenGrayscale())

        done = self.ale.game_over()
        return self.obs, reward, done

    def reset(self):
        self.ale.reset_game()
        self.obs = np.squeeze(self.ale.getScreenGrayscale())
        return self.obs

    def render(self, rate=0.1):
        self.im = Image.fromarray(self.obs)
        self.tkim = ImageTk.PhotoImage(self.im)
        self.window.configure(image=self.tkim)
        self.window.image = self.tkim
        self.window.update_idletasks()
        self.window.update()
        time.sleep(rate)

    def sample_action(self):
        action = random.choice([0, 1, 2, 3])
        return action
