import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import glob
from PIL import Image

class LidDrivenCavity():
    def __init__(self, length = 1, rho = 10, mu = 0.01, u = 1, iter = 2000, dt = 0.001, n_p = 101, iter_poisson = 50, gif = False) -> None:
        #Define parameters
        self.length = length
        self.rho = rho
        self.mu = mu
        self.u = u
        self.iterations = iter
        self.dt = dt
        self.n_p = n_p
        self.iterations_poisson = iter_poisson

        self.el_length = self.length / (self.n_p-1)

        self.result = None

        #GIF settings
        self.gif = gif
        self.each_iteration_figure = self.iterations/80

    def print_reynolds(self) -> None:
        print((self.u * self.length * self.rho) / self.mu)

    def central_difference_x(self, vec):
        result = np.zeros_like(vec)
        result[1:-1, 1:-1] = (vec[1:-1, 2:]-vec[1:-1, 0:-2]) / (2*self.el_length)
        return result
    
    def central_difference_y(self, vec):
        result = np.zeros_like(vec)
        result[1:-1, 1:-1] = (vec[2:, 1:-1] - vec[0:-2, 1:-1]) / (2*self.el_length)
        return result

    def laplace(self, vec):
        result = np.zeros_like(vec)
        result[1:-1, 1:-1] = (vec[1:-1, 0:-2] + vec[0:-2, 1:-1] - 4 * vec[1:-1, 1:-1] + vec[1:-1, 2:] + vec[2:, 1:-1]) / (self.el_length**2)
        return result

    def calculations(self):

        if self.gif == True:
            iteration_counter = 0
            number_of_image = 0

        x = np.linspace(0.0, self.length, self.n_p)
        y = np.linspace(0.0, self.length, self.n_p)

        X, Y = np.meshgrid(x, y)

        u_previous = np.zeros_like(X)
        v_previous = np.zeros_like(X)
        p_previous = np.zeros_like(X)
        
        for ite in tqdm(range(self.iterations)):
            d_u_previous_dx = self.central_difference_x(u_previous)
            d_u_previous_dy = self.central_difference_y(u_previous)
            d_v_previous_dx = self.central_difference_x(v_previous)
            d_v_previous_dy = self.central_difference_y(v_previous)
            laplace_u_previous = self.laplace(u_previous)
            laplace_v_previous = self.laplace(v_previous)

            u_tent = (u_previous + self.dt * (-(u_previous * d_u_previous_dx + v_previous * d_u_previous_dy) + self.mu * laplace_u_previous))
            v_tent = (v_previous + self.dt * (-(u_previous * d_v_previous_dx + v_previous * d_v_previous_dy) + self.mu * laplace_v_previous))

            u_tent[0, :] = 0
            u_tent[:, 0] = 0
            u_tent[:, -1] = 0
            u_tent[-1, :] = self.u
            v_tent[0, :] = 0
            v_tent[:, 0] = 0
            v_tent[:, -1] = 0
            v_tent[-1, :] = 0


            d_u_tent_dx = self.central_difference_x(u_tent)
            d_v_tent_dy = self.central_difference_y(v_tent)

            rhs = (self.rho / self.dt * (d_u_tent_dx + d_v_tent_dy))

            for _ in range(self.iterations_poisson):
                p_next = np.zeros_like(p_previous)
                p_next[1:-1, 1:-1] = 1/4 * (p_previous[1:-1, 0:-2] + p_previous[0:-2, 1:-1] + p_previous[1:-1, 2:] + p_previous[2:, 1:-1] - self.el_length**2 * rhs[1:-1, 1:-1])

                p_next[:, -1] = p_next[:, -2]
                p_next[0,  :] = p_next[1,  :]
                p_next[:,  0] = p_next[:,  1]
                p_next[-1, :] = 0.0

                p_previous = p_next
            

            d_p_next_dx = self.central_difference_x(p_next)
            d_p_next_dy = self.central_difference_y(p_next)

            u_next = (u_tent - self.dt / self.rho * d_p_next_dx)
            v_next = (v_tent - self.dt / self.rho * d_p_next_dy)

            u_next[0, :] = 0
            u_next[:, 0] = 0
            u_next[:, -1] = 0
            u_next[-1, :] = self.u
            v_next[0, :] = 0
            v_next[:, 0] = 0
            v_next[:, -1] = 0
            v_next[-1, :] = 0

            u_previous = u_next
            v_previous = v_next
            p_previous = p_next

            if self.gif == True:
                if iteration_counter == self.each_iteration_figure - 1:
                    iteration_counter = 0
                    figures_to_GIF(u_next, v_next, p_next, X, Y, ite, number_of_image)
                    number_of_image += 1
                    
                else:
                    iteration_counter += 1
                
        self.result = u_next, v_next, p_next, X, Y

def plotting(u, v, p, X, Y):

    fig, axes = plt.subplots(3, 1, figsize=(6, 12))

    c0 = axes[0].contourf(X[::2, ::2], Y[::2, ::2], u[::2, ::2], cmap="jet")
    axes[0].quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2], color="black")
    axes[0].set_xlim((0, 1))
    axes[0].set_ylim((0, 1))
    axes[0].set_title('U velocity')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_aspect('equal')
    axes[0].set_xticks(np.linspace(0, 1, 5))
    axes[0].set_yticks(np.linspace(0, 1, 5))
    axes[0].grid()

    c1 = axes[1].contourf(X[::2, ::2], Y[::2, ::2], v[::2, ::2], cmap="jet")
    axes[1].set_xlim((0, 1))
    axes[1].set_ylim((0, 1))
    axes[1].set_title('V velocity')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_aspect('equal')
    axes[1].set_xticks(np.linspace(0, 1, 5))
    axes[1].set_yticks(np.linspace(0, 1, 5))
    axes[1].grid()

    c2 = axes[2].contourf(X[::2, ::2], Y[::2, ::2], p[::2, ::2], cmap="jet")
    axes[2].streamplot(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2], color="black", linewidth=0.5)
    axes[2].set_xlim((0, 1))
    axes[2].set_ylim((0, 1))
    axes[2].set_title('Pressure')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].set_aspect('equal')
    axes[2].set_xticks(np.linspace(0, 1, 5))
    axes[2].set_yticks(np.linspace(0, 1, 5))
    axes[2].grid()

    cbar0 = fig.colorbar(c0, ax=axes[0])
    cbar1 = fig.colorbar(c1, ax=axes[1])
    cbar2 = fig.colorbar(c2, ax=axes[2])

    # plt.colorbar(ax=axes)
    plt.tight_layout()
    plt.show()

def figures_to_GIF(u, v, p, X, Y, ite, number_of_image):

    levels_1 = np.linspace(-0.3, 1.05, 10)
    levels_2 = np.linspace(-0.45, 0.29, 10)
    levels_3 = np.linspace(-10, 15, 10)

    plt.ioff()
    fig = plt.figure(figsize=(6, 4))
    plt.contourf(X[::2, ::2], Y[::2, ::2], u[::2, ::2], levels = levels_1, cmap="jet")
    plt.colorbar()
    plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2], color="black")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title('U velocity\nIteration = {}'.format(ite+1))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('figures/u/fig_{:05d}.png'.format(number_of_image))
    plt.close(fig)

    plt.ioff()
    fig = plt.figure(figsize=(6, 4))
    plt.contourf(X[::2, ::2], Y[::2, ::2], v[::2, ::2], levels = levels_2, cmap="jet")
    plt.colorbar()
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title('V velocity\nIteration = {}'.format(ite+1))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('figures/v/fig_{:05d}.png'.format(number_of_image))
    plt.close(fig)


    plt.ioff()
    fig = plt.figure(figsize=(6, 4))
    plt.contourf(X[::2, ::2], Y[::2, ::2], p[::2, ::2], levels = levels_3,cmap="jet")
    plt.colorbar()
    plt.streamplot(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2], color="black", linewidth=0.5)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title('Pressure\nIteration = {}'.format(ite+1))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('figures/p/fig_{:05d}.png'.format(number_of_image))
    plt.close(fig)

def make_GIF( frame_folder, name):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.png")]
    frame_one = frames[0]
    frame_one.save("GIFs/{}.gif".format(name), format="GIF", append_images=frames,
            save_all=True, duration=500, loop=0)