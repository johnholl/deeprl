from PIL import Image
import numpy as np


def preprocess(obs):
    img = Image.fromarray(obs)
    img = img.resize((84,84))
    new_obs = np.array(img)
    return new_obs
