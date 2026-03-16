import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc, multivariate_normal
from torch.utils.data import TensorDataset, DataLoader
import torch
from pathlib import Path

# ---------------------------
# Lorenz-96 two-timescale model definition
# ---------------------------
def lorenz96_two_scale(t, state, K, J, F, h, c, b):
    """
    Lorenz-96 two-timescale system.
    
    Parameters:
    - K: number of slow (X) variables
    - J: number of fast (Y) variables per X variable
    - F: forcing on X variables
    - h: coupling coefficient (Y -> X)
    - c: time-scale ratio
    - b: forcing on Y variables
    
    State vector: [X_0, ..., X_{K-1}, Y_{0,0}, ..., Y_{K-1,J-1}]
    Total state dimension: K + K*J
    """
    X = state[:K]
    Y = state[K:].reshape(K, J)
    
    # X dynamics
    dXdt = np.zeros(K)
    for k in range(K):
        dXdt[k] = -X[k-1] * (X[k-2] - X[(k+1) % K]) - X[k] + F - (h*c/b) * np.sum(Y[k, :])
    
    # Y dynamics
    dYdt = np.zeros((K, J))
    for k in range(K):
        for j in range(J):
            dYdt[k, j] = -c * b * Y[k, (j+1) % J] * (Y[k, (j+2) % J] - Y[k, j-1]) - c * Y[k, j] + (h*c/b) * X[k]
    
    return np.concatenate([dXdt, dYdt.ravel()])

# ---------------------------
# Sampling functions (4 physical parameters only)
# ---------------------------
def sample_parameters_sobol(N, F_range, h_range, c_range, b_range):
    """Sample 4D parameters using Sobol sequence."""
    sampler = qmc.Sobol(d=4, scramble=True)
    samples = sampler.random(N)
    
    lower_bounds = np.array([F_range[0], h_range[0], c_range[0], b_range[0]])
    upper_bounds = np.array([F_range[1], h_range[1], c_range[1], b_range[1]])
    
    scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)
    return scaled_samples

def sample_parameters_gaussian(N, mu, cov, F_range=None, h_range=None, c_range=None, b_range=None, clip=True):
    """
    Sample 4D parameters from Gaussian distribution.
    
    Args:
        N: Number of samples
        mu: Mean vector (4D) for [F, h, c, b]
        cov: Covariance matrix (4x4)
        *_range: Optional ranges for clipping
        clip: Whether to clip to ranges
    """
    samples = multivariate_normal(mean=mu, cov=cov).rvs(N)
    if samples.ndim == 1:  # Handle single sample case
        samples = samples.reshape(1, -1)
    
    # Clip to reasonable ranges if provided
    if clip:
        if F_range is not None:
            samples[:, 0] = np.clip(samples[:, 0], *F_range)
        if h_range is not None:
            samples[:, 1] = np.clip(samples[:, 1], *h_range)
        if c_range is not None:
            samples[:, 2] = np.clip(samples[:, 2], *c_range)
        if b_range is not None:
            samples[:, 3] = np.clip(samples[:, 3], *b_range)
    
    return samples

# ---------------------------
# Dataset creation
# ---------------------------
def generate_l96_dataset(
    prior_type="gaussian",
    N_params=1000,
    N_traj_per_param=10,
    T_window=1000,
    dt=0.01,
    discard_time=0.1,
    K=8,
    J=8,
    # Parameter ranges
    F_range=(5, 20),
    h_range=(0.5, 2.0),
    c_range=(5, 15),
    b_range=(8, 12),
    # Gaussian prior parameters
    mu=(10, 1.0, 10, 10),
    cov=None,
    clip=True,
):
    """
    Generate L96 two-timescale dataset.
    
    Args:
        prior_type: 'sobol' or 'gaussian'
        N_params: Number of parameter sets to sample
        N_traj_per_param: Number of trajectories per parameter set
        T_window: Number of time steps to save
        dt: Time step for integration
        discard_time: Initial time to discard for transient
        K: Number of slow (X) variables
        J: Number of fast (Y) variables per X
        *_range: Parameter ranges [min, max]
        mu: Mean for Gaussian prior (4D): [F, h, c, b]
        cov: Covariance for Gaussian prior (4x4)
        clip: Whether to clip sampled parameters to ranges
    
    Returns:
        params_arr: (N_total, 4) array of parameters [F, h, c, b]
        X_arr: (N_total, T, K) array of slow trajectories
        Y_arr: (N_total, T, K*J) array of fast trajectories
    """
    # Default covariance if not provided
    if cov is None:
        cov = np.diag([5**2, 0.5**2, 2**2, 1**2])
    
    # Sample parameters
    if prior_type == "sobol":
        params_set = sample_parameters_sobol(N_params, F_range, h_range, c_range, b_range)
    elif prior_type == "gaussian":
        params_set = sample_parameters_gaussian(
            N_params, mu, cov, 
            F_range, h_range, c_range, b_range, 
            clip=clip
        )
    else:
        raise ValueError(f"Unknown prior type: {prior_type}. Use 'sobol' or 'gaussian'.")
    
    print ('discard_time:', discard_time)
    # Simulate trajectories
    t_eval_full = np.arange(0, discard_time + T_window*dt, dt)
    X_trajectories = []
    Y_trajectories = []
    params_records = []
    
    for idx, (F, h, c, b) in enumerate(params_set):
        for _ in range(N_traj_per_param):
            # Random initial conditions for X and Y
            init_X = np.random.uniform(-5, 5, size=K)
            init_Y = np.random.uniform(-5, 5, size=K*J)
            init_state = np.concatenate([init_X, init_Y])
            
            # Integrate ODEs
            sol = solve_ivp(
                lorenz96_two_scale, 
                (0, t_eval_full[-1]), 
                init_state,
                args=(K, J, F, h, c, b), 
                t_eval=t_eval_full, 
                method='RK45',
                rtol=1e-6, 
                atol=1e-8
            )
            
            # Extract trajectories after discard period
            start_idx = int(discard_time/dt)
            end_idx = start_idx + T_window
            
            X_traj = sol.y[:K, start_idx:end_idx].T
            Y_traj = sol.y[K:, start_idx:end_idx].T
            
            X_trajectories.append(X_traj)
            Y_trajectories.append(Y_traj)
            params_records.append([F, h, c, b])
    
    params_arr = np.array(params_records)
    X_arr = np.array(X_trajectories)
    Y_arr = np.array(Y_trajectories)
    
    return params_arr, X_arr, Y_arr

# ---------------------------
# Function to generate and save dataset
# ---------------------------
def create_and_save_dataset(
    save_path,
    prior_type="gaussian",
    N_params=200,
    N_traj_per_param=50,
    T_window=1000,
    dt_subsample=5,
    test_split=0.2,
    K=8,
    J=8,
    # Parameter ranges
    F_range=(5, 20),
    h_range=(0.5, 2.0),
    c_range=(5, 15),
    b_range=(8, 12),
    # Gaussian prior
    mu=(10, 1.0, 10, 10),
    cov=None,
    clip=True,
    avg=False,
    seed=42
):
    """
    Generate and save L96 dataset.
    
    Args:
        save_path: Path to save NPZ file
        prior_type: 'sobol' or 'gaussian'
        N_params: Number of parameter sets
        N_traj_per_param: Trajectories per parameter set
        T_window: Time steps to save
        dt_subsample: Subsampling factor for time
        test_split: Fraction for test set
        K: Number of slow variables
        J: Fast variables per slow variable
        *_range: Parameter ranges
        mu: Mean for Gaussian prior [F, h, c, b]
        cov: Covariance matrix (4x4)
        clip: Whether to clip parameters to ranges
        avg: Whether to average over time
        seed: Random seed
    
    Saves:
        params_train/test: (N, 4) parameters [F, h, c, b]
        X_train/test: (N, T, K) slow variables
        Y_train/test: (N, T, K*J) fast variables
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Generate full dataset
    params, X_trajs, Y_trajs = generate_l96_dataset(
        prior_type=prior_type,
        N_params=N_params,
        N_traj_per_param=N_traj_per_param,
        T_window=T_window,
        K=K,
        J=J,
        F_range=F_range,
        h_range=h_range,
        c_range=c_range,
        b_range=b_range,
        mu=mu,
        cov=cov,
        clip=clip,
    )
    
    # Subsample or average timesteps
    if avg:
        X_trajs = np.mean(X_trajs, axis=1, keepdims=True)
        Y_trajs = np.mean(Y_trajs, axis=1, keepdims=True)
    else:
        X_trajs = X_trajs[:, ::dt_subsample, :]
        Y_trajs = Y_trajs[:, ::dt_subsample, :]
    
    # Shuffle & split
    total_samples = len(params)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    split_idx = int(total_samples * (1 - test_split))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    params_train, params_test = params[train_idx], params[test_idx]
    X_train, X_test = X_trajs[train_idx], X_trajs[test_idx]
    Y_train, Y_test = Y_trajs[train_idx], Y_trajs[test_idx]
    
    # Save as .npz
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        params_train=params_train, X_train=X_train, Y_train=Y_train,
        params_test=params_test, X_test=X_test, Y_test=Y_test
    )
    
    print(f"Dataset saved to {save_path}")
    print(f"  Training samples: {len(train_idx)}")
    print(f"  Test samples: {len(test_idx)}")
    print(f"  X shape: {X_train.shape}, Y shape: {Y_train.shape}")
    print(f"  Parameters shape: {params_train.shape}")

# ---------------------------
# Simple NumPy-based DataLoader
# ---------------------------
class Lorenz96DataLoader:
    def __init__(self, params, X_trajs, Y_trajs, batch_size=16, shuffle=True):
        self.parameters = params
        self.X_trajectories = X_trajs
        self.Y_trajectories = Y_trajs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.parameters))
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for start_idx in range(0, len(self.parameters), self.batch_size):
            batch_idx = self.indices[start_idx:start_idx+self.batch_size]
            yield (self.X_trajectories[batch_idx], 
                   self.Y_trajectories[batch_idx], 
                   self.parameters[batch_idx])
    
    def __len__(self):
        return int(np.ceil(len(self.parameters) / self.batch_size))

# ---------------------------
# Function to load dataset into train/test loaders
# ---------------------------
def load_dataset_loaders(
    path, 
    batch_size=128, 
    dt=1, 
    T_window=1000,
    avg=False, 
    seed=42,
    return_both_XY=True
):
    """
    Load L96 dataset with reproducible train/test split and ordering.
    
    Args:
        path: Path to NPZ file
        batch_size: Batch size for DataLoaders
        dt: Subsampling rate for time dimension
        T_window: Maximum time window to use
        avg: If True, average over time dimension
        seed: Random seed for reproducibility
        return_both_XY: If True, return concatenated [X, Y]; if False, return only X
    
    Returns:
        train_loader, test_loader
        Each batch contains (trajectories, parameters) where:
        - trajectories: (batch, T, dim) with dim=K or K+K*J depending on return_both_XY
        - parameters: (batch, 4) for [F, h, c, b]
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load data from NPZ
    data_loaded = np.load(path)
    data = dict(data_loaded)
    data_loaded.close()
    
    # Cut trajectories to T_window
    data["X_train"] = data["X_train"][:, :T_window, :]
    data["X_test"] = data["X_test"][:, :T_window, :]
    data["Y_train"] = data["Y_train"][:, :T_window, :]
    data["Y_test"] = data["Y_test"][:, :T_window, :]
    
    # Subsample timesteps or average
    if avg:
        data["X_train"] = np.mean(data["X_train"], axis=1, keepdims=True)
        data["X_test"] = np.mean(data["X_test"], axis=1, keepdims=True)
        data["Y_train"] = np.mean(data["Y_train"], axis=1, keepdims=True)
        data["Y_test"] = np.mean(data["Y_test"], axis=1, keepdims=True)
    else:
        data["X_train"] = data["X_train"][:, ::dt, :]
        data["X_test"] = data["X_test"][:, ::dt, :]
        data["Y_train"] = data["Y_train"][:, ::dt, :]
        data["Y_test"] = data["Y_test"][:, ::dt, :]
    
    # Prepare trajectories
    X_train = torch.tensor(data['X_train'], dtype=torch.float32)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32)
    Y_train = torch.tensor(data['Y_train'], dtype=torch.float32)
    Y_test = torch.tensor(data['Y_test'], dtype=torch.float32)
    
    # Combine X and Y if requested
    if return_both_XY:
        C_train = torch.cat([X_train, Y_train], dim=-1)
        C_test = torch.cat([X_test, Y_test], dim=-1)
    else:
        C_train = X_train
        C_test = X_test
    
    # Prepare parameters (4D: F, h, c, b)
    Params_train = torch.tensor(data['params_train'], dtype=torch.float32)
    Params_test = torch.tensor(data['params_test'], dtype=torch.float32)
    
    print(f"Loaded dataset:")
    print(f"  Train: {C_train.shape} trajectories, {Params_train.shape} parameters")
    print(f"  Test:  {C_test.shape} trajectories, {Params_test.shape} parameters")
    
    # Create TensorDatasets
    train_dataset = TensorDataset(C_train, Params_train)
    test_dataset = TensorDataset(C_test, Params_test)
    
    # Create generator for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # Worker seed function
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        worker_init_fn=seed_worker,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader
