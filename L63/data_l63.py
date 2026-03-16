import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc, multivariate_normal
from torch.utils.data import TensorDataset, DataLoader
import torch
from pathlib import Path
# ---------------------------
# Lorenz-63 model definition
# ---------------------------
def lorenz63(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# ---------------------------
# Sampling functions
# ---------------------------
def sample_parameters_sobol(N, sigma_range, rho_range, beta_range):
    sampler = qmc.Sobol(d=3, scramble=True)
    samples = sampler.random(N)

    lower_bounds = np.array([sigma_range[0], rho_range[0], beta_range[0]])
    upper_bounds = np.array([sigma_range[1], rho_range[1], beta_range[1]])

    scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)
    return scaled_samples

def sample_parameters_gaussian(N, mu, cov, sigma_range, rho_range, beta_range):
    samples = multivariate_normal(mean=mu, cov=cov).rvs(N)
    # Clip to reasonable ranges
    samples[:, 0] = np.clip(samples[:, 0], *sigma_range)
    samples[:, 1] = np.clip(samples[:, 1], *rho_range)
    samples[:, 2] = np.clip(samples[:, 2], *beta_range)
    return samples



def sample_parameters_mixture(N, mus, covs, weights, ranges):
    weights = np.array(weights) / np.sum(weights)
    components = np.random.choice(len(weights), size=N, p=weights)
    samples = np.zeros((N, len(mus[0])))

    for i, comp in enumerate(components):
        samples[i] = multivariate_normal(mean=mus[comp], cov=covs[comp]).rvs()

    # for dim, (low, high) in enumerate(ranges):
    #     samples[:, dim] = np.clip(samples[:, dim], low, high)

    return samples


# ----- Dataset creation -----
def generate_l63_dataset(
    prior_type="sobol",
    N_params=1000,
    N_traj_per_param=10,
    T_window=100,
    dt=0.01,
    discard_time=10.0,
    sigma_range=(5, 20),
    rho_range=(20, 40),
    beta_range=(2, 4),
    mu=(10, 28, 8/3),
    cov=None,
    mus=None,
    covs=None,
    weights=None
):
    if cov is None:
        cov = np.diag([2**2, 5**2, 0.3**2])

    # Sample parameters based on prior type
    if prior_type == "sobol":
        params_set = sample_parameters_sobol(N_params, sigma_range, rho_range, beta_range)
    elif prior_type == "gaussian":
        params_set = sample_parameters_gaussian(N_params, mu, cov, sigma_range, rho_range, beta_range)
    elif prior_type == "mixture_gaussian":
        if mus is None or covs is None or weights is None:
            raise ValueError("For mixture_gaussian, provide mus, covs, and weights.")
        ranges = [sigma_range, rho_range, beta_range]
        params_set = sample_parameters_mixture(N_params, mus, covs, weights, ranges)
    else:
        raise ValueError("Unknown prior type. Use 'sobol', 'gaussian', or 'mixture_gaussian'.")

    t_eval_full = np.arange(0, discard_time + T_window*dt, dt)

    trajectories = []
    parameters = []

    for sigma, rho, beta in params_set:
        for _ in range(N_traj_per_param):
            init_state = np.random.uniform(-15, 15, size=3)
            sol = solve_ivp(lorenz63, (0, t_eval_full[-1]), init_state,
                            args=(sigma, rho, beta), t_eval=t_eval_full,
                            method='RK45')
            traj = sol.y[:, int(discard_time/dt): int(discard_time/dt) + T_window].T
            trajectories.append(traj)
            parameters.append([sigma, rho, beta])

    return np.array(parameters), np.array(trajectories)


# ---------------------------
# Function to generate and save dataset
# ---------------------------
def create_and_save_dataset(
    save_path,
    prior_type="sobol",
    N_params=200,
    N_traj_per_param=50,
    T_window=100,
    dt_subsample=5,
    test_split=0.2,
    sigma_range=(5, 20),
    rho_range=(20, 40),
    beta_range=(2, 4),
    mu=(10, 28, 8/3),
    cov=None,
    mus=None,
    covs=None,
    weights=None
):
    # Generate full dataset
    parameters, trajectories = generate_l63_dataset(
        prior_type=prior_type,
        N_params=N_params,
        N_traj_per_param=N_traj_per_param,
        T_window=T_window,
        sigma_range=sigma_range,
        rho_range=rho_range,
        beta_range=beta_range,
        mu=mu,
        cov=cov,
        mus=mus,
        covs=covs,
        weights=weights
    )
    
    # Subsample timesteps
    trajectories = trajectories[:, ::dt_subsample, :]
    
    # Shuffle indices
    total_samples = len(parameters)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    # Split into train/test
    split_idx = int(total_samples * (1 - test_split))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    params_train, trajs_train = parameters[train_idx], trajectories[train_idx]
    params_test, trajs_test = parameters[test_idx], trajectories[test_idx]
    
    # Save as .npz
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        params_train=params_train, trajs_train=trajs_train,
        params_test=params_test, trajs_test=trajs_test
    )
    print(f"Dataset saved to {save_path}")


# ---------------------------
# Simple NumPy-based DataLoader
# ---------------------------
class Lorenz63DataLoader:
    def __init__(self, params, trajs, batch_size=16, shuffle=True):
        self.parameters = params
        self.trajectories = trajs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.parameters))
        
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for start_idx in range(0, len(self.parameters), self.batch_size):
            batch_idx = self.indices[start_idx:start_idx+self.batch_size]
            yield self.trajectories[batch_idx], self.parameters[batch_idx]
            
    def __len__(self):
        return int(np.ceil(len(self.parameters) / self.batch_size))

# ---------------------------
# Function to load dataset into train/test loaders
# ---------------------------

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def load_dataset_loaders(path, batch_size=128, seed=42):
    """
    Load dataset with reproducible train/test split and ordering.
    
    Args:
        path: Path to NPZ file
        batch_size: Batch size for DataLoaders
        seed: Random seed for reproducibility
    
    Returns:
        train_loader, test_loader
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load data from NPZ
    data = np.load(path)
    
    # Extract variables from file
    C_train = torch.tensor(data['trajs_train'], dtype=torch.float32)
    X_train = torch.tensor(data['params_train'], dtype=torch.float32)
    C_test  = torch.tensor(data['trajs_test'], dtype=torch.float32)
    X_test  = torch.tensor(data['params_test'], dtype=torch.float32)
    
    # Create TensorDatasets
    train_dataset = TensorDataset(C_train, X_train)
    test_dataset  = TensorDataset(C_test, X_test)
    
    # Create generator for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # Worker seed function for multi-process data loading
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
    
    # Create DataLoaders with reproducibility settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        generator=generator,        # For reproducible shuffling
        worker_init_fn=seed_worker, # For reproducible workers
        num_workers=0               # Set to 0 for full reproducibility
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False  # Keep False for test set
    )
    
    return train_loader, test_loader


def load_dataset_loaders_old(path, batch_size=128):
    # Load data from NPZ
    data = np.load(path)
    
    # Extract variables from file (update keys if needed)
    C_train = torch.tensor(data['trajs_train'], dtype=torch.float32)
    X_train = torch.tensor(data['params_train'], dtype=torch.float32)
    C_test  = torch.tensor(data['trajs_test'], dtype=torch.float32)
    X_test  = torch.tensor(data['params_test'], dtype=torch.float32)

    # Create TensorDatasets
    train_dataset = TensorDataset(C_train, X_train)
    test_dataset  = TensorDataset(C_test, X_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
