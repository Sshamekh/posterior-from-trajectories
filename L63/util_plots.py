import matplotlib.pyplot as plt
import numpy as np
import corner


def plot_parameter_scatter(x_true, x_samples,run_name,save = False):
    """
    Plots histograms of true vs predicted parameters for all features.
    
    Args:
        x_true: numpy array of true parameters, shape (N, x_dim)
        x_samples: numpy array of samples, shape (N, n_samples, x_dim) or (n_samples, N, x_dim)
    """
    param_name = ['sigma', 'rho', 'betta']
    # Ensure shape is (N, n_samples, x_dim)
    if x_samples.shape[0] != x_true.shape[0]:
        x_samples = np.transpose(x_samples, (1, 0, 2))

    x_pred_mean = np.mean(x_samples, axis=1)  # mean over samples

    n_features = x_true.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(5*n_features, 4))

    

    axes[0].scatter(x_pred_mean[:, 1], x_pred_mean[:, 2],alpha=0.5, label='Predicted')
    axes[0].scatter(x_true[:, 1],x_true[:, 2],  alpha=0.5, label='True')

    axes[0].set_ylabel('beta')
    axes[0].set_xlabel('rho')
    axes[0].legend()


    axes[1].scatter(x_pred_mean[:, 0], x_pred_mean[:, 2],alpha=0.5, label='Predicted')
    axes[1].scatter(x_true[:, 0],x_true[:, 2],  alpha=0.5, label='True')
    
    axes[1].set_ylabel('beta')
    axes[1].set_xlabel('sigma')
    
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    if save:
        fig.savefig(f"./plots/{run_name}_scatter.jpeg",bbox_inches = 'tight',dpi  =100)


def plot_parameter_histograms(x_true, x_samples,run_name,save = False, theta_params = False):
    """
    Plots histograms of true vs predicted parameters for all features.
    
    Args:
        x_true: numpy array of true parameters, shape (N, x_dim)
        x_samples: numpy array of samples, shape (N, n_samples, x_dim) or (n_samples, N, x_dim)
    """
    param_name = ['sigma', 'rho', 'betta']
    param_name = ['theta_1', 'theta_2', 'theta_3', 'theta_4', 'theta_5']
    # Ensure shape is (N, n_samples, x_dim)
    if x_samples.shape[0] != x_true.shape[0]:
        x_samples = np.transpose(x_samples, (1, 0, 2))

    x_pred_mean = np.mean(x_samples, axis=1)  # mean over samples
    x_pred_mean = x_samples[:,10,:]
    n_features = x_true.shape[1]
    fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 4))

    if n_features == 1:
        axes = [axes]  # ensure iterable if only one feature

    for i in range(1,n_features):
        axes[i].hist(x_true[:, i], bins=50, alpha=0.5, label='True', density=True)
        axes[i].hist(x_pred_mean[:, i], bins=50, alpha=0.5, label='Predicted', density=True)
        axes[i].set_title(f'Parameter: {param_name[i]}', fontsize=15)
        axes[i].legend()
        axes[i].tick_params(axis='both', which='major', labelsize=14)
        axes[i].set_xlim(-0.5,1.5)


    if theta_params:
        axes[0].hist((x_true[:, 1] + x_true[:, 0])/2, bins=50, alpha=0.5, label='True', density=True)
        axes[0].hist((x_pred_mean[:, 1] + x_pred_mean[:, 0])/2, bins=50, alpha=0.5, label='True', density=True)
        axes[0].tick_params(axis='both', which='major', labelsize=14)
        axes[0].set_title(f'Parameter: {param_name[0]}', fontsize=15)
        axes[0].set_xlim(-0.5,1.5)

        
    plt.tight_layout()
    plt.show()
    if save:
        fig.savefig(f"./plots/{run_name}_hist.jpeg",bbox_inches = 'tight',dpi  =100)


def compute_coverage_analysis(models_dict, key, credible_levels=[0.5, 0.68, 0.90, 0.95]):
    """
    Compute empirical coverage for different credible intervals
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary containing params_true and params_generated
    key : str
        Key to access the model data
    credible_levels : list
        List of credible levels to compute (e.g., 0.95 for 95% CI)
    
    Returns:
    --------
    coverage_results : dict
        Dictionary with coverage statistics for each parameter
    """
    params_true = models_dict[key]["params_true"]
    params_generated = models_dict[key]["params_generated"]
    
    n_trajectories, n_samples, n_params = params_generated.shape
    
    # Find unique true parameter sets
    unique_params, inverse_indices = np.unique(params_true, axis=0, return_inverse=True)
    n_unique = len(unique_params)
    
    # Storage for coverage results
    coverage_results = {level: np.zeros((n_unique, n_params)) for level in credible_levels}
    
    # For each unique parameter set
    for idx in range(n_unique):
        # Find all trajectories with this parameter set
        traj_indices = np.where(inverse_indices == idx)[0]
        
        # For each parameter dimension
        for param_idx in range(n_params):
            true_value = unique_params[idx, param_idx]
            
            # Collect posterior samples from all trajectories with same true params
            all_posterior_samples = params_generated[traj_indices, :, param_idx].flatten()
            
            # Compute credible intervals
            for level in credible_levels:
                alpha = (1 - level) / 2
                lower = np.percentile(all_posterior_samples, alpha * 100)
                upper = np.percentile(all_posterior_samples, (1 - alpha) * 100)
                
                # Check if true value is within interval
                coverage_results[level][idx, param_idx] = (lower <= true_value <= upper)
    
    # Compute empirical coverage (fraction of times true value is in interval)
    empirical_coverage = {level: np.mean(coverage_results[level], axis=0) 
                          for level in credible_levels}
    
    return empirical_coverage, credible_levels


def plot_calibration(empirical_coverage, credible_levels, param_labels):
    """
    Plot calibration curve showing empirical vs nominal coverage
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    
    n_params = len(param_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, n_params))
    
    # Plot diagonal (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
    
    # Plot each parameter
    for param_idx in range(n_params):
        coverages = [empirical_coverage[level][param_idx] for level in credible_levels]
        ax.plot(credible_levels, coverages, 'o-', linewidth=2, markersize=8,
                color=colors[param_idx], label=param_labels[param_idx])
    
    ax.set_xlabel('Nominal Credible Level', fontsize=12)
    ax.set_ylabel('Empirical Coverage', fontsize=12)
    ax.set_title('Calibration Plot: Coverage Analysis', fontsize=14)
    ax.legend(fontsize=10,loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1.1])
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    return fig


def plot_parameter_recovery(models_dict, key, param_labels):
    """
    Plot true vs predicted parameters for all trajectory-parameter pairs
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary containing params_true and params_generated
    key : str
        Key to access the model data
    param_labels : list
        List of parameter names
    
    Returns:
    --------
    fig : matplotlib figure
    """
    params_true = models_dict[key]["params_true"]
    params_generated = models_dict[key]["params_generated"]
    
    n_trajectories, n_samples, n_params = params_generated.shape
    
    # Compute posterior mean for each trajectory
    params_predicted_mean = np.mean(params_generated, axis=1)  # shape: (n_trajectories, n_params)
    
    # Compute posterior std for color coding
    params_predicted_std = np.std(params_generated, axis=1)  # shape: (n_trajectories, n_params)
    
    # Create subplots
    fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 5))
    
    if n_params == 1:
        axes = [axes]
    
    for i, (ax, label) in enumerate(zip(axes, param_labels)):
        # Scatter plot: true vs predicted
        scatter = ax.scatter(params_true[:, i], params_predicted_mean[:, i], 
                           c=params_predicted_std[:, i], 
                           alpha=0.6, s=20, cmap='viridis', edgecolors='none')
        
        # Add diagonal line (perfect recovery)
        min_val = min(params_true[:, i].min(), params_predicted_mean[:, i].min())
        max_val = max(params_true[:, i].max(), params_predicted_mean[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                label='Perfect recovery', alpha=0.7)
        
        # Labels and formatting
        ax.set_xlabel(f'True {label}', fontsize=12)
        ax.set_ylabel(f'Predicted {label} (mean)', fontsize=12)
        ax.set_title(f'{label}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])

        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02,aspect=15, shrink=0.7)
        cbar.set_label('Posterior Std', fontsize=10)
        
        # Compute and display R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(params_true[:, i], params_predicted_mean[:, i])
        ax.text(0.05, 0.95, f'$R^2$ = {r2:.3f}', 
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def compare_posteriors(nflow_samples, true_samples, labels=None, 
                      truths=None,  figsize=(10, 10)):
    """
    Compare NFlow posterior with true Bayesian posterior.
    
    Args:
        nflow_samples: samples from normalizing flow (n_samples, n_params)
        true_samples: samples from MCMC/true Bayesian (n_samples, n_params)
        labels: parameter names
        truths: true parameter values
    """
    if labels is None:
        labels = ['σ', 'ρ', 'β']
    
    # Create figure
    fig = corner.corner(
        true_samples,
        labels=labels,
        color='red',
        alpha=0.6,
        hist_kwargs={'density': True, 'alpha': 0.6},
        label='true density',
        truths=truths,
        truth_color='black',
        show_titles=False,
        smooth=1.0,
        levels=(0.68, 0.95),  # 1σ and 2σ contours
        plot_datapoints=False,  # Don't show individual points
        fill_contours=True,
        range=[(-0.3, 1.3) for _ in labels],
        figsize=figsize
    )
    
    # Overlay NFlow samples
    corner.corner(
        nflow_samples,
        fig=fig,
        color='blue',
        alpha=0.6,
        hist_kwargs={'density': True, 'alpha': 0.6},
        label='NFlow estimation',
        smooth=1.0,
        levels=(0.68, 0.95),
        plot_datapoints=False,
        fill_contours=True,
        range=[(-0.3, 1.3) for _ in labels]
    )
    
    # Add legend
    axes = np.array(fig.axes).reshape((len(labels), len(labels)))
    axes[0, -1].legend(loc='upper right', fontsize=12, frameon=True)
    

def compare_posteriors_same_params(params , labels, truths,   **kwargs):
    """
    Compare posterior distributions from multiple trajectories
    
    params1, params2: arrays of shape (n_samples, n_params)
    labels: list of parameter names
    truths: true parameter values
    """
    n_params = len(labels)
    fig, axes = plt.subplots(1, n_params, figsize=(4*n_params, 4))
    n_cases = params.shape[0]
    if n_params == 1:
        axes = [axes]

    for i, (ax, label) in enumerate(zip(axes, labels)):
        for j in range(n_cases):
            ax.hist(params[j,:, i], bins=30, alpha=0.3, label=f'Case {j+1}', density=True)
        
        # Add vertical line for true value
        ax.axvline(truths[i], color='red', linestyle='--', linewidth=2, label='True value')
        
        ax.set_xlabel(label)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig
