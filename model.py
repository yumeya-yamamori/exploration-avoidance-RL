import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import imageio
import shutil

import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection

from scipy.special import softmax


class GridWorld:
    def __init__(self, width=10, length=None):
        self.width = width
        if length is None:
            self.length = self.width
        else:
            self.length = length

        # State space (the grid world, including off-platform locations)
        self.states = np.arange(self.length * self.width)

        # Grid shape
        self.grid = np.ones((self.length, self.width))

        # Extrinsic rewards
        self.rewards = np.zeros((self.length, self.width))

    # Set model parameters
    def set_parameters(
        self,
        gamma=0.95,
        beta_int=1,
        tau=0.5,
        beta_ext_pos=1,
        beta_ext_neg=1,
    ):
        self.gamma = abs(gamma)
        self.beta_int = abs(beta_int)
        self.tau = abs(tau)
        self.beta_ext_pos = abs(beta_ext_pos)
        self.beta_ext_neg = abs(beta_ext_neg)

        # Check values of parameters
        if self.gamma < 0 or self.gamma >= 1:
            raise ValueError("Gamma should be contained in (0, 1).")
        if self.tau == 0:
            raise ValueError(
                "Exploration discount parameter should be strictly positive."
            )

        # Scale extrinsic rewards with weights
        self.scaled_rewards = self.rewards.copy()
        self.scaled_rewards[self.scaled_rewards > 0] *= self.beta_ext_pos
        self.scaled_rewards[self.scaled_rewards < 0] *= self.beta_ext_neg

        # Get legal action set
        rows, cols = self.legal_states.shape

        # Define action indices (down, up, left, right, stay)
        actions = np.arange(5)

        # Get legal actions for each position
        self.all_legal_actions = [[[] for _ in range(cols)] for _ in range(rows)]

        for x in range(cols):
            for y in range(rows):
                # Check for illegal actions based on state location
                if self.legal_states[y, x] == 0:
                    legal_actions = [
                        False
                    ] * 5  # Assume all actions are initially legal
                else:
                    legal_actions = [True] * 5  # Assume all actions are initially legal

                    if y == 0:
                        legal_actions[0] = False
                    elif self.legal_states[y - 1, x] == 0:
                        legal_actions[0] = False

                    if y == rows - 1:
                        legal_actions[1] = False
                    elif self.legal_states[y + 1, x] == 0:
                        legal_actions[1] = False

                    if x == 0:
                        legal_actions[2] = False
                    elif self.legal_states[y, x - 1] == 0:
                        legal_actions[2] = False

                    if x == cols - 1:
                        legal_actions[3] = False
                    elif self.legal_states[y, x + 1] == 0:
                        legal_actions[3] = False

                    # Assign the list of legal actions for the current state
                    self.all_legal_actions[y][x] = actions[legal_actions]

        # Get the transition matrix
        self.legal_indices = np.where(self.legal_states.flatten())[0]
        N_legal_states = len(self.legal_indices)
        self.T = np.eye(N_legal_states)

        for ix in range(N_legal_states):

            # Convert to 2D index
            sx, sy = np.unravel_index(self.legal_indices[ix], self.grid.shape)

            # Check legal DOWN action
            if sy > 0:
                if self.legal_states[sx, sy - 1] == 1:
                    s_next = np.ravel_multi_index((sx, sy - 1), self.grid.shape)

                    self.T[ix, self.legal_indices == s_next] = 1

            # Check legal UP action
            if sy < self.grid.shape[1] - 1:
                if self.legal_states[sx, sy + 1] == 1:
                    s_next = np.ravel_multi_index((sx, sy + 1), self.grid.shape)
                    self.T[ix, self.legal_indices == s_next] = 1

            # Check legal LEFT action
            if sx > 0:
                if self.legal_states[sx - 1, sy] == 1:
                    s_next = np.ravel_multi_index((sx - 1, sy), self.grid.shape)
                    self.T[ix, self.legal_indices == s_next] = 1

            # Check legal RIGHT action
            if sx < self.grid.shape[0] - 1:
                if self.legal_states[sx + 1, sy] == 1:
                    s_next = np.ravel_multi_index((sx + 1, sy), self.grid.shape)
                    self.T[ix, self.legal_indices == s_next] = 1

            # Normalize
            self.T[ix, :] /= self.T[ix, :].sum()

    # Simulate exploration data
    def simulate(self, s0_x=None, s0_y=None, nsteps=100):
        # Start in centre as default
        if s0_x is None:
            s0_x = self.width // 2
        if s0_y is None:
            s0_y = self.length // 2

        # Compute the transition probability matrix for legal states
        I = np.eye(self.legal_states.sum())
        M = np.linalg.inv(I - self.gamma * self.T)
        M *= 1 - self.gamma

        state = self.convert_state(r=s0_y, c=s0_x)
        walk = int(state)

        # Tracker of visited states
        state_visits = np.zeros(self.grid.shape)

        # State visit rewards
        r = (state_visits + 1) ** -self.tau

        for _ in range(nsteps - 1):

            sy, sx = self.convert_state(state=state)

            # Terminate if in illegal state
            if self.legal_states[sy, sx] == 0:
                break

            # Mark the current state as visited
            state_visits[sy, sx] += 1
            r[sy, sx] = (state_visits[sy, sx] + 1) ** -self.tau

            # Map of rewards = intrinsic + extrinsic
            R = self.beta_int * r + self.scaled_rewards  # !!!

            # Get legal actions
            legal_actions = self.all_legal_actions[sy][sx]
            Qs = np.zeros(len(legal_actions))

            for k in range(len(legal_actions)):
                next_state = self.transition(state, legal_actions[k])
                Qs[k] = sum(
                    R[self.legal_states == 1]
                    * M[self.legal_indices == next_state, :][0]
                )

            # Softmax choice probabilities
            pa = softmax(Qs)
            a = np.random.choice(legal_actions, p=pa)

            # Make transition and save
            state = self.transition(state, a)
            walk = np.append(walk, int(state))

        self.walk = walk

    # Just plot the legal positions/states
    def plot_legal_states(
        self,
        ax=None,
        annot_states=False,
        state_alpha=0.5,
    ):
        # If no Axes object is passed, create one
        if ax is None:
            ax = plt.gca()

        # Plot the matrix using imshow
        if np.unique(self.legal_states).__len__() == 1:
            cmap = ListedColormap(["none"])  # 'none' is for transparency
        else:
            cmap = ListedColormap(["black", "none"])  # 'none' is for transparency
        sns.heatmap(self.legal_states, cmap=cmap, cbar=False, square=True)

        ax.set_xlim([0, self.width])
        ax.set_ylim([0, self.length])

        # Get the dimensions of the heatmap
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Create a rectangle patch with the same size as the heatmap
        rect = patches.Rectangle(
            (xlim[0], ylim[0]),
            xlim[1] - xlim[0],
            ylim[1] - ylim[0],
            linewidth=2,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(rect)

        # Annotate state IDs if required
        if annot_states is True:
            # Overlay index numbers on each cell
            for index, value in enumerate(self.states):
                # Calculate the 2D coordinates from the 1D index
                row, col = divmod(index, self.grid.shape[1])
                # Overlay the index number with alpha=0.5
                plt.text(
                    col + 1 / 2,
                    row + 1 / 2,
                    str(index),
                    color="black",
                    ha="center",
                    va="center",
                    alpha=state_alpha,
                )

        return ax

    # Plot state value map
    def plot_features(self, ax=None, cbar=True, vmin=None, vmax=None, alpha=1.0):
        # If no Axes object is passed, create one
        if ax is None:
            ax = plt.gca()

        # Plot colour only if we have variable values
        if np.all(np.logical_or(np.isnan(self.rewards), self.rewards == 0.0)):
            alpha = 0.0

        # Show platform
        sns.heatmap(
            self.rewards.copy().reshape(self.grid.shape),
            cbar=cbar,
            cbar_kws={"label": "V"},
            cmap=sns.diverging_palette(20, 220, as_cmap=True),
            vmin=vmin,
            vmax=vmax,
            center=0,
            alpha=alpha,
            ax=ax,
            square=True,
        )

        ax.set_xlim([0, self.width])
        ax.set_ylim([0, self.length])

        return ax

    # Plot the environment with its features
    def plot_world(
        self,
        ax=None,
        annot_states=False,
        state_alpha=0.5,
    ):
        ax = self.plot_legal_states(ax, annot_states, state_alpha)

        return ax

    def plot_walk(self, ax=None, smooth=5, show_legend=True, cmap=None):
        # If no Axes object is passed, create one
        if ax is None:
            ax = plt.gca()
        if cmap is None:
            cmap = "winter"

        # Get Cartesian co-ordinates of smoothed movement
        y, x = self.convert_state(state=self.walk)
        x = self.moving_average(x, smooth)
        y = self.moving_average(y, smooth)

        # Show starting position
        ax.scatter(
            x[0] + 0.5,
            y[0] + 0.5,
            c="black",
            alpha=0.9,
            s=20,
            marker="x",
            label="Starting\nposition",
        )

        # Show walk
        points = np.array([x + 1 / 2, y + 1 / 2]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=plt.get_cmap(cmap), alpha=0.9)
        lc.set_array(np.arange(len(x)))
        lc.set_linewidth(2)
        ax.add_collection(lc)

        # Legend
        if show_legend:
            ax.legend(loc="upper right")

        return ax

    def plot_position(self, state, ax=None):
        # If no Axes object is passed, create one
        if ax is None:
            ax = plt.gca()

        # Get Cartesian co-ordinates of smoothed movement
        y, x = self.convert_state(state=state)

        # Show starting position
        ax.scatter(
            x + 0.5,
            y + 0.5,
            c="black",
            alpha=0.9,
            s=20,
        )

        return ax

    def plot_visits(self, state_visits, ax=None, alpha=1.0):
        # If no Axes object is passed, create one
        if ax is None:
            ax = plt.gca()

        # Create a mask for 0 visits
        mask = state_visits.reshape(self.grid.shape) == 0

        # Show platform
        sns.heatmap(
            state_visits.reshape(self.grid.shape),
            cmap=sns.dark_palette("seagreen", as_cmap=True),
            cbar_kws={"label": "Visits"},
            ax=ax,
            square=True,
            mask=mask,
            alpha=alpha,
        )

        ax.set_xlim([0, self.width])
        ax.set_ylim([0, self.length])

        return ax

    def walk_gif(self, dur=0.01, save_name="tmp", show_features=True):

        # Base directory for all GIFs
        base_gif_dir = "animations"
        os.makedirs(base_gif_dir, exist_ok=True)

        # Temporary directory for current animation frames
        temp_dir = os.path.join(base_gif_dir, "temp_" + save_name)
        os.makedirs(temp_dir, exist_ok=True)

        # Plot past visits
        state_visits = np.zeros(self.grid.shape)
        filenames = []

        for i in range(len(self.walk)):

            # Get state
            state = self.walk[i]

            # Update and plot
            state_visits[self.convert_state(state=state)] += 1
            f = self.plot_world()
            if show_features:
                self.plot_features(ax=f)
            self.plot_visits(state_visits, ax=f, alpha=0.5)
            self.plot_position(state=state, ax=f)

            # Save frame
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            plt.savefig(frame_path)
            plt.close()
            filenames.append(frame_path)

        # Create PNG of walk map
        save_file_path = os.path.join(base_gif_dir, save_name + ".png")
        f = self.plot_walk(smooth=5)
        self.plot_world(ax=f)
        plt.savefig(save_file_path)

        # Create GIF
        save_file_path = os.path.join(base_gif_dir, save_name + ".gif")
        with imageio.get_writer(
            save_file_path, mode="I", duration=dur, loop=0
        ) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        # Cleanup: Remove temporary directory and its contents
        shutil.rmtree(temp_dir)

    def convert_state(self, state=None, r=None, c=None):
        if state is not None:
            # Convert single number to co-ordinates
            r = state // self.width  # Integer division to find the row index
            c = state % self.width  # Modulo operation to find the column index
            return r, c
        else:
            # Convert co-ordinates to a single number
            state = r * self.width + c
            return state

    def states_to_actions(sx, sy):
        if len(sx) < 2 or len(sy) < 2:
            raise ValueError("State index lengths must be greater than 1.")
        elif len(sx) != len(sy):
            raise ValueError("State index lengths must be equal.")

        actions = np.zeros(len(sx), dtype=int)
        actions[-1] = 4

        for t in range(len(sx) - 1):
            if sx[t] == sx[t + 1] and sy[t] == (sy[t + 1] - 1):
                actions[t] = 0  # Down
            elif sx[t] == sx[t + 1] and sy[t] == (sy[t + 1] + 1):
                actions[t] = 1  # Up
            elif sx[t] == (sx[t + 1] - 1) and sy[t] == sy[t + 1]:
                actions[t] = 2  # Left
            elif sx[t] == (sx[t + 1]) + 1 and sy[t] == sy[t + 1]:
                actions[t] = 3  # Right
            elif sx[t] == sx[t + 1] and sy[t] == sy[t + 1]:
                actions[t] = 4  # Stay
            else:
                raise ValueError(
                    f"Invalid state transition from ({sx[t]}, {sy[t]}) to ({sx[t+1]}, {sy[t+1]})"
                )

        return actions

    def transition(self, state, action):
        # DOWN
        if action == 0:
            state -= self.width
        # UP
        elif action == 1:
            state += self.width
        # LEFT
        elif action == 2:
            state -= 1
        # RIGHT
        elif action == 3:
            state += 1

        return state

    def invert_transition(self, state0, state1):
        # DOWN
        if state1 - state0 == -self.width:
            action = 0
        # UP
        elif state1 - state0 == +self.width:
            action = 1
        # LEFT
        elif state1 - state0 == -1:
            action = 2
        # RIGHT
        elif state1 - state0 == +1:
            action = 3
        # REMAIN
        else:
            action = 4

        return action

    def check_transition(self, state, action):

        # Get co-ordinates
        r, c = self.convert_state(state=state)

        # Check for borders
        if action == 0:
            r -= 1
        elif action == 1:
            r += 1
        elif action == 2:
            c -= 1
        elif action == 3:
            c += 1

        # Return false if border reached
        if r < 0 or c < 0 or r >= self.length or c >= self.width:
            return False
        # Check for illegal state
        elif self.legal_states[r, c] == 0:
            return False
        # Legal otherwise
        else:
            return True

    @staticmethod
    def moving_average(x, window=5):
        x = np.cumsum(x, dtype=float)
        x[window:] = x[window:] - x[:-window]
        return x[window - 1 :] / window

    def heatmap(self, values, ax=None, cmap=None, cbar_name=None):
        # If no Axes object is passed, create one
        if ax is None:
            ax = plt.gca()
        if cmap is None:
            cmap = sns.diverging_palette(20, 220, as_cmap=True)
            center_cmap = 0
        else:
            center_cmap = None
        if cbar_name is None:
            cbar_name = ""

        # Create a mask for 0 values
        mask = values.reshape(self.grid.shape) == 0

        # Show platform
        sns.heatmap(
            values.reshape(self.grid.shape),
            cmap=cmap,
            center=center_cmap,
            cbar_kws={"label": cbar_name},
            ax=ax,
            square=True,
            mask=mask,
            alpha=0.9,
        )

        ax.set_xlim([0, self.width])
        ax.set_ylim([0, self.length])

        return ax


class SandBox(GridWorld):
    def __init__(self, width=15):

        # Force shape to be a square with odd length sides
        if width % 2 == 0:
            width += 1
        length = width

        super().__init__(width, length)  # Initialize the base class

        # Legal positions
        self.legal_states = np.ones((self.length, self.width), dtype="int")

        # Add aversion to inner region
        self.rewards = np.zeros(self.grid.shape)


class LightDarkBox(GridWorld):
    def __init__(self, width=16, length=11):

        # Force width to be of some form: 3 * x + 1
        if (width - 1) % 3 != 0:
            width += 3 - ((width - 1) % 3)
        # Force length to be odd
        if length % 2 == 0:
            length += 1

        super().__init__(width, length)  # Initialize the base class

        # Legal positions
        self.legal_states = np.ones((self.length, self.width), dtype="int")
        self.legal_states[:, (self.width - 1) // 3] = 0
        self.legal_states[self.length // 2, (self.width - 1) // 3] = 1

        # Add aversion to lit region
        self.rewards[:, (self.width - 1) // 3 + 1 :] = -1


class OpenFieldTest(GridWorld):
    def __init__(self, width=15):

        # Force shape to be a square with odd length sides
        if width % 2 == 0:
            width += 1
        length = width

        super().__init__(width, length)  # Initialize the base class

        # Legal positions
        self.legal_states = np.ones((self.length, self.width), dtype="int")

        # Add aversion to inner region
        self.rewards = -1 * np.minimum(
            self.width // 2
            - abs(np.tile(np.arange(self.width) - self.width // 2, (self.length, 1))),
            self.length // 2
            - abs(
                np.tile(np.arange(self.length) - self.length // 2, (self.width, 1))
            ).T,
        )

        # Normalise to [0, 1]
        self.rewards = self.rewards / abs(self.rewards.min())


class ElevatedPlusMaze(GridWorld):
    def __init__(self, width=35, arm_width=5):

        # Force shape to be a square with odd length sides
        if width % 2 == 0:
            width += 1
        length = width

        # Force arm width to be odd
        if arm_width % 2 == 0:
            arm_width += 1

        super().__init__(width, length)  # Initialize the base class

        self.arm_width = arm_width

        # Legal positions
        self.legal_states = np.zeros((self.length, self.width), dtype="int")
        self.legal_states[
            :, (self.width - self.arm_width) // 2 : (self.width + self.arm_width) // 2
        ] = 1
        self.legal_states[
            (self.length - self.arm_width) // 2 : (self.length + self.arm_width) // 2, :
        ] = 1

        # Add aversion to open arms
        self.rewards
        self.rewards[
            : (self.width - self.arm_width) // 2,
            (self.width - self.arm_width) // 2 : (self.width + self.arm_width) // 2,
        ] = -1
        self.rewards[
            (self.width + self.arm_width) // 2 :,
            (self.width - self.arm_width) // 2 : (self.width + self.arm_width) // 2,
        ] = -1
