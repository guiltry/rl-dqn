import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps


class Utils:

    @staticmethod
    def show_image(prev_state, next_state, action, reward):
        fig, axs = plt.subplots(1, 8, figsize=(16, 4))
        for ax in axs:
            ax.axis('off')

        n = len(prev_state)
        for i in range(n):
            plt.subplot(1, 8, i + 1)
            plt.imshow(Image.fromarray(np.reshape(prev_state[i], (125, 175))), cmap='gray')

        for i in range(n):
            plt.subplot(1, 8, i + 1 + n)
            plt.imshow(Image.fromarray(np.reshape(next_state[i], (125, 175))), cmap='gray')

        plt.tight_layout()
        plt.show()
        print(f'Action = {action}, Reward = {reward}')
#%%
