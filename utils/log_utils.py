import os
import tempfile
from datetime import datetime

import absl.flags as flags
import ml_collections
import numpy as np
import jax
import jax.numpy as jnp
import wandb
from PIL import Image, ImageEnhance


class CsvLogger:
    """CSV logger for logging metrics to a CSV file."""

    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)

    def log(self, row, step):
        row['step'] = step
        if self.file is None:
            self.file = open(self.path, 'w')
            if self.header is None:
                self.header = [k for k, v in row.items() if not isinstance(v, self.disallowed_types)]
                self.file.write(','.join(self.header) + '\n')
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        else:
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


def get_exp_name(seed):
    """Return the experiment name."""
    exp_name = ''
    exp_name += f'sd{seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    exp_name += f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    return exp_name


def get_flag_dict():
    """Return the dictionary of flags."""
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS if '.' not in k}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict


def setup_wandb(
    entity=None,
    project='project',
    group=None,
    name=None,
    mode='online',
):
    """Set up Weights & Biases for logging."""
    wandb_output_dir = tempfile.mkdtemp()
    tags = [group] if group is not None else None

    init_kwargs = dict(
        config=get_flag_dict(),
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        name=name,
        settings=wandb.Settings(
            start_method='thread',
            _disable_stats=False,
        ),
        mode=mode,
        save_code=True,
    )

    run = wandb.init(**init_kwargs)

    return run


def reshape_video(v, n_cols=None):
    """Helper function to reshape videos."""
    if v.ndim == 4:
        v = v[None,]

    _, t, h, w, c = v.shape

    if n_cols is None:
        # Set n_cols to the square root of the number of videos.
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, newshape=(t, c, n_rows * h, n_cols * w))

    return v


def get_wandb_video(renders=None, n_cols=None, fps=15):
    """Return a Weights & Biases video.

    It takes a list of videos and reshapes them into a single video with the specified number of columns.

    Args:
        renders: List of videos. Each video should be a numpy array of shape (t, h, w, c).
        n_cols: Number of columns for the reshaped video. If None, it is set to the square root of the number of videos.
    """
    # Pad videos to the same length.
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        assert render.dtype == np.uint8

        # Decrease brightness of the padded frames.
        final_frame = render[-1]
        final_image = Image.fromarray(final_frame)
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(0.5)
        final_frame = np.array(final_image)

        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
        renders[i] = np.concatenate([render, pad], axis=0)

        # Add borders.
        renders[i] = np.pad(renders[i], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    renders = np.array(renders)  # (n, t, h, w, c)

    renders = reshape_video(renders, n_cols)  # (t, c, nr * h, nc * w)

    return wandb.Video(renders, fps=fps, format='mp4')

def xy_to_ij(env, xy):
    maze_unit = env.unwrapped._maze_unit
    i = ((xy[1] + env.unwrapped._offset_y + 0.5 * maze_unit) / maze_unit).astype(int)
    j = ((xy[0] + env.unwrapped._offset_x + 0.5 * maze_unit) / maze_unit).astype(int)
    return i, j

def ij_to_xy(env, ij):
    i, j = ij
    x = j * env.unwrapped._maze_unit - env.unwrapped._offset_x
    y = i * env.unwrapped._maze_unit - env.unwrapped._offset_y
    return x, y

def check_valid(env, coords):
    maze_map = jnp.array(env.unwrapped.maze_map)
    ij = xy_to_ij(env, coords)
    return maze_map[ij[0], ij[1]]

def get_grid(maze_env, density=1):
    maze_map = jnp.array(maze_env.unwrapped.maze_map)
    # print("maze_map shape: ",maze_map.shape)
    upper_left_x, upper_left_y = ij_to_xy(maze_env, (0, 0))
    lower_right_x, lower_right_y = ij_to_xy(maze_env, maze_map.shape)
    x = jnp.linspace(upper_left_x, lower_right_x, int(lower_right_x - upper_left_x) * density)
    y = jnp.linspace(upper_left_y, lower_right_y, int(lower_right_y - upper_left_y) * density)
    x, y = jnp.meshgrid(x, y)
    # print("meshgrid shape: ",x.shape)
    x, y= x.ravel(), y.ravel()
    return jnp.stack([x, y], axis=-1)

def get_valid_coords(env, density=1):
    grid = get_grid(env, density)
    # print(grid.shape)
    ij = jax.vmap(check_valid, in_axes=(None, 0))(env, grid)
    valid = grid[jnp.argwhere(ij ==0),...]
    return valid.reshape(-1, 2)

def evaluate_value(
    agent,
    env,
    goal,
    density=1,
    actions=None,
    use_action=False
):
    "returns a grid of values that corresponds to the initial state"
    "value is calculated as d^{mrn}_theta(s_0, g)"
    valid_coords = get_valid_coords(env, density)
    observation = env.unwrapped.get_ob()
    original_obs = observation[:2]

    all_obs = np.repeat(observation[None], valid_coords.shape[0], axis=0)
    # print("all_obs.shape", all_obs.shape)
    all_obs[:, :2] = valid_coords
    # if all_obs.shape[1] == 4:
    #     all_obs[:, 2:] = np.random.randn(all_obs.shape[0], 2)*0.1
    
    all_obs = jnp.array(all_obs)
    goal = jnp.repeat(goal[None], valid_coords.shape[0], axis=0)
    if use_action:
        key = jax.random.PRNGKey(np.random.randint(0, 1000000))
        split = jax.random.split(key, goal.shape[0])
        actions = jax.vmap(agent.sample_actions, in_axes=(0, 0, 0))(all_obs, goal, split)
    else:
        actions = None
    dist = agent.get_distance(all_obs, goal, actions)
    # phi = agent.network.select('psi')(all_obs)
    # psi = agent.network.select('psi')(goal)
    # dist = agent.mrn_distance(phi, psi)
    return dist, original_obs

def log_distances(agent, env, num_tasks, density=1, actions=None, use_action=False):
    dists = []
    goals = []
    original_obs = []
    coords = get_valid_coords(env, density)
    for i in range(1, num_tasks + 1):
        _, info = env.reset(options=dict(task_id=i, render_goal=True))
        goal = info['goal']
        dist, original_ob = evaluate_value(agent, env, goal, density, actions, use_action)
        dist = jnp.max(dist, axis=0)
        dists.append(dist)
        goals.append(goal)
        original_obs.append(original_ob)
    return dists, goals, coords, original_obs

