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

def sample_parameters_mixture(N, mus, covs, weights, ranges=None):
    weights = np.array(weights) / np.sum(weights)
    components = np.random.choice(len(weights), size=N, p=weights)
    samples = np.zeros((N, len(mus[0])))

    for i, comp in enumerate(components):
        samples[i] = multivariate_normal(mean=mus[comp], cov=covs[comp]).rvs()

    # If ranges are given, optionally clip (kept off by default for latent)
    if ranges is not None:
        for dim, (low, high) in enumerate(ranges):
            samples[:, dim] = np.clip(samples[:, dim], low, high)
    return samples

# ---------------------------
# NEW: Latent (overparameterized) samplers and mapping
# ---------------------------
# --- helpers unchanged in your file ---
def map_latent_to_physical(thetas):
    sigma = thetas[:, 2]
    rho   = (thetas[:, 0] + thetas[:, 1])/2.
    beta  = thetas[:, 3]
    return np.stack([sigma, rho, beta], axis=-1)

def sample_latent_gaussian(N, mu_L, cov_L):
    return multivariate_normal(mean=mu_L, cov=cov_L).rvs(N)

def sample_latent_mixture(N, mus_L, covs_L, weights_L):
    return sample_parameters_mixture(N, mus_L, covs_L, weights_L, ranges=None)

# NEW: physical->latent stub so that map_latent_to_physical(latent_stub) == physical
def physical_to_latent_stub(params_set_3d):
    """
    params_set_3d: array (N,3) in order [sigma, rho, beta]
    Return latent (N,5) with mapping:
      theta1 = rho, theta2 = 0, theta3 = sigma, theta4 = beta, theta5 = 0
    so that map_latent_to_physical recovers the same (sigma,rho,beta).
    """
    N = params_set_3d.shape[0]
    theta = np.zeros((N, 5))
    theta[:, 0] = params_set_3d[:, 1]   # rho
    theta[:, 1] = 0.0
    theta[:, 2] = params_set_3d[:, 0]   # sigma
    theta[:, 3] = params_set_3d[:, 2]   # beta
    theta[:, 4] = 0.0
    return theta


def map_latent_to_physical(thetas):
    """
    Map latent theta = [theta1,...,theta5] to physical (sigma, rho, beta):
        rho   = theta1 + theta2
        sigma = theta3
        beta  = theta4
    theta5 is unused (dummy).
    Returns array of shape (N, 3) in order [sigma, rho, beta].
    """
    sigma = thetas[:, 2]
    rho   = thetas[:, 0] + thetas[:, 1]
    beta  = thetas[:, 3]
    return np.stack([sigma, rho, beta], axis=-1)

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
    # physical-space Gaussian defaults
    mu=(10, 28, 8/3),
    cov=None,
    mus=None,
    covs=None,
    weights=None,
    # latent-space prior controls
    mu_L=None,
    cov_L=None,
    mus_L=None,
    covs_L=None,
    weights_L=None,
    # control clipping of physical parameters after mapping
    clip_physical=False,
    # NEW: optionally also return physical (sigma,rho,beta) for sanity checks
    return_physical=False,
):
    """
    prior_type options:
      - 'sobol'                      : QMC over (sigma,rho,beta)
      - 'gaussian'                   : 3D Gaussian over (sigma,rho,beta)
      - 'mixture_gaussian'           : 3D mixture over (sigma,rho,beta)
      - 'latent_gaussian_diag'       : 5D Gaussian over theta with diagonal cov
      - 'latent_gaussian_corr'       : 5D Gaussian over theta with full cov
      - 'latent_mixture_gaussian'    : 5D mixture over theta
    Returns:
      theta_arr : (N_total, 5)  -> saved later as theta_train/theta_test
      trajs_arr : (N_total, T, 3)
      (optional) phys_arr : (N_total, 3) in order [sigma, rho, beta]
    """
    if cov is None:
        cov = np.diag([2**2, 5**2, 0.3**2])
    if mu_L is None:
        mu_L = np.zeros(5)
    if cov_L is None:
        cov_L = np.diag([5.0, 5.0, 5.0, 1.0, 1.0])

    # 1) Sample either physical params (3D) or latent (5D)
    if prior_type == "sobol":
        params_set = sample_parameters_sobol(N_params, sigma_range, rho_range, beta_range)
        latent_set = physical_to_latent_stub(params_set)

    elif prior_type == "gaussian":
        params_set = sample_parameters_gaussian(N_params, mu, cov, sigma_range, rho_range, beta_range)
        latent_set = physical_to_latent_stub(params_set)

    elif prior_type == "mixture_gaussian":
        if mus is None or covs is None or weights is None:
            raise ValueError("For mixture_gaussian, provide mus, covs, and weights.")
        params_set = sample_parameters_mixture(N_params, mus, covs, weights, ranges=None)
        if clip_physical:
            params_set[:, 0] = np.clip(params_set[:, 0], *sigma_range)
            params_set[:, 1] = np.clip(params_set[:, 1], *rho_range)
            params_set[:, 2] = np.clip(params_set[:, 2], *beta_range)
        latent_set = physical_to_latent_stub(params_set)

    elif prior_type in ("latent_gaussian_diag", "latent_gaussian_corr"):
        if prior_type == "latent_gaussian_diag":
            cov_L = np.diag(np.diag(cov_L))
        latent_set = sample_latent_gaussian(N_params, mu_L, cov_L)
        params_set = map_latent_to_physical(latent_set)
        if clip_physical:
            params_set[:, 0] = np.clip(params_set[:, 0], *sigma_range)
            params_set[:, 1] = np.clip(params_set[:, 1], *rho_range)
            params_set[:, 2] = np.clip(params_set[:, 2], *beta_range)

    elif prior_type == "latent_mixture_gaussian":
        if mus_L is None or covs_L is None or weights_L is None:
            raise ValueError("For latent_mixture_gaussian, provide mus_L, covs_L, and weights_L.")
        latent_set = sample_latent_mixture(N_params, mus_L, covs_L, weights_L)
        params_set = map_latent_to_physical(latent_set)
        if clip_physical:
            params_set[:, 0] = np.clip(params_set[:, 0], *sigma_range)
            params_set[:, 1] = np.clip(params_set[:, 1], *rho_range)
            params_set[:, 2] = np.clip(params_set[:, 2], *beta_range)

    else:
        raise ValueError("Unknown prior type.")

    # 2) Simulate trajectories using physical params derived from latent
    t_eval_full = np.arange(0, discard_time + T_window*dt, dt)
    trajectories = []
    theta_records = []
    phys_records  = [] if return_physical else None

    for idx, (sigma, rho, beta) in enumerate(params_set):
        for _ in range(N_traj_per_param):
            init_state = np.random.uniform(-15, 15, size=3)
            sol = solve_ivp(
                lorenz63, (0, t_eval_full[-1]), init_state,
                args=(sigma, rho, beta), t_eval=t_eval_full, method='RK45'
            )
            traj = sol.y[:, int(discard_time/dt): int(discard_time/dt) + T_window].T
            trajectories.append(traj)
            # Save THETA (5D) for this (sigma,rho,beta)
            theta_records.append(latent_set[idx])
            if phys_records is not None:
                phys_records.append([sigma, rho, beta])

    theta_arr = np.array(theta_records)
    trajs_arr = np.array(trajectories)
    if return_physical:
        phys_arr = np.array(phys_records)
        return theta_arr, trajs_arr, phys_arr
    return theta_arr, trajs_arr

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
    weights=None,
    # latent controls
    mu_L=None,
    cov_L=None,
    mus_L=None,
    covs_L=None,
    weights_L=None,
    clip_physical=False,
    # NEW: choose whether to save physical copies too
    save_physical=False,
    avg=False
):
    # Generate full dataset (always return theta; optionally physical)
    if save_physical:
        theta, trajectories, phys = generate_l63_dataset(
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
            weights=weights,
            mu_L=mu_L,
            cov_L=cov_L,
            mus_L=mus_L,
            covs_L=covs_L,
            weights_L=weights_L,
            clip_physical=clip_physical,
            return_physical=True,
        )
    else:
        theta, trajectories = generate_l63_dataset(
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
            weights=weights,
            mu_L=mu_L,
            cov_L=cov_L,
            mus_L=mus_L,
            covs_L=covs_L,
            weights_L=weights_L,
            clip_physical=clip_physical,
            return_physical=False,
        )

    # Subsample timesteps
    if avg:
        trajectories = np.mean(trajectories, axis=1, keepdims=True)
    else:
        trajectories = trajectories[:, ::dt_subsample, :]

    # Shuffle & split
    total_samples = len(theta)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    split_idx = int(total_samples * (1 - test_split))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    theta_train, theta_test = theta[train_idx], theta[test_idx]
    trajs_train, trajs_test = trajectories[train_idx], trajectories[test_idx]

    # Save as .npz
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    if save_physical:
        phys_train, phys_test = phys[train_idx], phys[test_idx]
        np.savez_compressed(
            save_path,
            theta_train=theta_train, trajs_train=trajs_train,
            theta_test=theta_test,   trajs_test=trajs_test,
            phys_train=phys_train,   phys_test=phys_test
        )
    else:
        np.savez_compressed(
            save_path,
            theta_train=theta_train, trajs_train=trajs_train,
            theta_test=theta_test,   trajs_test=trajs_test
        )
    print(f"Dataset saved to {save_path}")
# ---------------------------
# Simple NumPy-based DataLoader
# (unchanged; omitted for brevity if you had one below)

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

import numpy as np
import torch

def normalize_trajectories(X_train, Y_train, X_test, Y_test, method='standardize'):
    """
    Normalize X and Y trajectories separately.
    
    Args:
        X_train, Y_train: Training trajectories (N, T, K) and (N, T, K*J)
        X_test, Y_test: Test trajectories
        method: 'standardize', 'minmax', or 'robust'
    
    Returns:
        Normalized trajectories and statistics for denormalization
    """
    stats = {}
    
    if method == 'standardize':
        # Compute statistics over ALL dimensions (samples, time, features)
        stats['X_mean'] = X_train.mean()
        stats['X_std'] = X_train.std()
        stats['Y_mean'] = Y_train.mean()
        stats['Y_std'] = Y_train.std()
        
        # Normalize
        X_train_norm = (X_train - stats['X_mean']) / (stats['X_std'] + 1e-8)
        X_test_norm = (X_test - stats['X_mean']) / (stats['X_std'] + 1e-8)
        Y_train_norm = (Y_train - stats['Y_mean']) / (stats['Y_std'] + 1e-8)
        Y_test_norm = (Y_test - stats['Y_mean']) / (stats['Y_std'] + 1e-8)
    
    elif method == 'minmax':
        # Scale to [0, 1] or [-1, 1]
        stats['X_min'] = X_train.min()
        stats['X_max'] = X_train.max()
        stats['Y_min'] = Y_train.min()
        stats['Y_max'] = Y_train.max()
        
        X_train_norm = (X_train - stats['X_min']) / (stats['X_max'] - stats['X_min'] + 1e-8)
        X_test_norm = (X_test - stats['X_min']) / (stats['X_max'] - stats['X_min'] + 1e-8)
        Y_train_norm = (Y_train - stats['Y_min']) / (stats['Y_max'] - stats['Y_min'] + 1e-8)
        Y_test_norm = (Y_test - stats['Y_min']) / (stats['Y_max'] - stats['Y_min'] + 1e-8)
        
        # Optional: scale to [-1, 1] instead
        X_train_norm = 2 * X_train_norm - 1
        X_test_norm = 2 * X_test_norm - 1
        Y_train_norm = 2 * Y_train_norm - 1
        Y_test_norm = 2 * Y_test_norm - 1
    
    elif method == 'robust':
        # Use median and IQR (robust to outliers)
        stats['X_median'] = np.median(X_train)
        stats['X_iqr'] = np.percentile(X_train, 75) - np.percentile(X_train, 25)
        stats['Y_median'] = np.median(Y_train)
        stats['Y_iqr'] = np.percentile(Y_train, 75) - np.percentile(Y_train, 25)
        
        X_train_norm = (X_train - stats['X_median']) / (stats['X_iqr'] + 1e-8)
        X_test_norm = (X_test - stats['X_median']) / (stats['X_iqr'] + 1e-8)
        Y_train_norm = (Y_train - stats['Y_median']) / (stats['Y_iqr'] + 1e-8)
        Y_test_norm = (Y_test - stats['Y_median']) / (stats['Y_iqr'] + 1e-8)
    
    return X_train_norm, Y_train_norm, X_test_norm, Y_test_norm, stats

def load_dataset_loaders(
    path, 
    batch_size=128, 
    dt=1, 
    T_window=100, 
    start_time=0, 
    avg=False, 
    subsample_avg=False,
    seed=42,
    variables='both'  # 'X', 'Y', or 'both'
):
    """
    Load L96 dataset with reproducible train/test split and ordering.
    
    Args:
        path: Path to NPZ file
        batch_size: Batch size for DataLoaders
        dt: Subsampling rate for time dimension
        T_window: End time for trajectory window
        start_time: Start time for trajectory window
        avg: If True, average over time dimension
        seed: Random seed for reproducibility
        variables: Which variables to return
            - 'X': Only slow variables (shape: batch, T, K)
            - 'Y': Only fast variables (shape: batch, T, K*J)
            - 'both': Concatenated [X, Y] (shape: batch, T, K+K*J)
    
    Returns:
        train_loader, test_loader
        Each batch contains (trajectories, parameters) where:
        - trajectories: (batch, T, dim) depending on 'variables' argument
        - parameters: (batch, 4) for [F, h, c, b]
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load data from NPZ
    data_loaded = np.load(path)
    data = dict(data_loaded)  # Convert to mutable dictionary
    data_loaded.close() 

    # Cutting the trajectories to window [start_time:T_window]
    data["X_train"] = data["X_train"][:, start_time:T_window, :]
    data["X_test"]  = data["X_test"][:, start_time:T_window, :]
    data["Y_train"] = data["Y_train"][:, start_time:T_window, :]
    data["Y_test"]  = data["Y_test"][:, start_time:T_window, :]

    # Subsample timesteps or average
    if avg:
        data["X_train"] = np.mean(data["X_train"], axis=1, keepdims=True)
        data["X_test"]  = np.mean(data["X_test"], axis=1, keepdims=True)
        data["Y_train"] = np.mean(data["Y_train"], axis=1, keepdims=True)
        data["Y_test"]  = np.mean(data["Y_test"], axis=1, keepdims=True)
    elif subsample_avg:
        # Subsample by averaging over dt-sized windows
        def subsample_avg_func(arr, dt):
            N, T, D = arr.shape
            T_new = T // dt
            arr = arr[:, :T_new*dt, :].reshape(N, T_new, dt, D)
            # print ('T_new:', T_new, ' dt:', dt)
            return arr.mean(axis=2)
        
        data["X_train"] = subsample_avg_func(data["X_train"], dt)
        data["X_test"]  = subsample_avg_func(data["X_test"], dt)
        data["Y_train"] = subsample_avg_func(data["Y_train"], dt)
        data["Y_test"]  = subsample_avg_func(data["Y_test"], dt)
    else:
        data["X_train"] = data["X_train"][:, ::dt, :]
        data["X_test"]  = data["X_test"][:, ::dt, :]
        data["Y_train"] = data["Y_train"][:, ::dt, :]
        data["Y_test"]  = data["Y_test"][:, ::dt, :]

    X_train, Y_train, X_test, Y_test, _ = normalize_trajectories(data["X_train"], data["Y_train"],
                                                                 data["X_test"], data["Y_test"],
                                                                 method='standardize')
    print ('data shapes after normalization:', X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    # Select which variables to use based on 'variables' argument
    if variables == 'X':
        C_train = torch.tensor(X_train, dtype=torch.float32)
        C_test  = torch.tensor(X_test, dtype=torch.float32)
    elif variables == 'Y':
        C_train = torch.tensor(Y_train, dtype=torch.float32)
        C_test  = torch.tensor(Y_test, dtype=torch.float32)
    elif variables == 'both':
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test  = torch.tensor(X_test, dtype=torch.float32)
        Y_train = torch.tensor(Y_train, dtype=torch.float32)
        Y_test  = torch.tensor(Y_test, dtype=torch.float32)
        # Concatenate along feature dimension
        C_train = torch.cat([X_train, Y_train], dim=-1)
        C_test  = torch.cat([X_test, Y_test], dim=-1)
    else:
        raise ValueError(f"Invalid variables argument: '{variables}'. Must be 'X', 'Y', or 'both'.")

    # Load parameters (4D: F, h, c, b)
    X_train = torch.tensor(data['params_train'], dtype=torch.float32)
    X_test  = torch.tensor(data['params_test'], dtype=torch.float32)

    print(f"Loaded dataset with variables='{variables}':")
    print(f"  Parameters: {X_test.shape}")
    print(f"  Trajectories: {C_test.shape}")

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