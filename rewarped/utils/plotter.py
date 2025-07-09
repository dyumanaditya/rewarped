import numpy as np
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self):
        self.gradients = []
        self.initial_velocities = []
        self.loss = []

        self.gradient_labels = ['xd_grad', 'yd_grad']
        self.velocity_labels = ['xd_vel', 'yd_vel']

    def add_gradient(self, gradient):
        self.gradients.append(gradient)

    def add_initial_velocity(self, initial_velocity):
        self.initial_velocities.append(initial_velocity)

    def add_loss(self, loss):
        self.loss.append(loss)

    def plot(self):
        print('plotting')
        gradients = np.array(self.gradients)
        velocities = np.array(self.initial_velocities)
        losses = np.array(self.loss)

        fig, axs = plt.subplots(3, 1, figsize=(10, 12))

        axs[0].plot(gradients[:, 0], label=self.gradient_labels[0])
        axs[0].plot(gradients[:, 1], label=self.gradient_labels[1])
        axs[0].set_ylim(-30, 60)
        axs[0].set_xlabel('Iteration')
        axs[0].set_ylabel('Gradient Value')
        axs[0].set_title('Gradient Evolution Over Time')
        axs[0].legend()

        axs[1].plot(velocities[:, 0], label=self.velocity_labels[0])
        axs[1].plot(velocities[:, 1], label=self.velocity_labels[1])
        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel('Velocity Value')
        axs[1].set_title('Initial Velocity Evolution Over Time')
        axs[1].legend()

        axs[2].plot(losses, label='Loss')
        axs[2].set_xlabel('Iteration')
        axs[2].set_ylabel('Loss Value')
        axs[2].set_title('Loss Evolution Over Time')
        axs[2].legend()

        plt.tight_layout()
        plt.show()
