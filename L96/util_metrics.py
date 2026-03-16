import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from typing import Dict
class PosteriorMetrics:
    """
    Comprehensive metrics for evaluating normalizing flow posterior estimates
    """
    
    def __init__(self, param_names=['sigma', 'rho', 'beta']):
        """
        Initialize with parameter names
        
        Args:
            param_names: List of parameter names (default: Lorenz 63)
        """
        self.param_names = param_names
        self.n_params = len(param_names)
        
    def compute_all_metrics(self, 
                           theta_true: np.ndarray,
                           posterior_samples: np.ndarray,
                           prior_samples: np.ndarray = None,
                           log_probs: np.ndarray = None) -> Dict:
        """
        Compute all metrics for a test set
        
        Args:
            theta_true: (N_test, n_params) - true parameter values
            posterior_samples: (N_test, N_samples, n_params) - posterior samples from flow
            prior_samples: (N_prior, n_params) - samples from training prior (optional)
            log_probs: (N_test,) - log probability of true params under posterior (optional)
            
        Returns:
            Dictionary with all computed metrics
        """
        N_test = theta_true.shape[0]
        N_samples = posterior_samples.shape[1]
        
        print(f"Computing metrics for {N_test} test trajectories with {N_samples} samples each...")
        
        metrics = {}
        
        # ============================================================
        # 1. POINT ESTIMATION METRICS
        # ============================================================
        print("  Computing point estimation metrics...")
        
        # Posterior mean as point estimate
        theta_mean = posterior_samples.mean(axis=1)  # (N_test, n_params)
        
        # Overall metrics
        metrics['rmse_overall'] = np.sqrt(mean_squared_error(
            theta_true.flatten(), theta_mean.flatten()
        ))
        metrics['mae_overall'] = mean_absolute_error(
            theta_true.flatten(), theta_mean.flatten()
        )
        
        # Per-parameter metrics
        metrics['rmse_per_param'] = {}
        metrics['mae_per_param'] = {}
        metrics['bias_per_param'] = {}
        metrics['r2_per_param'] = {}
        
        for i, param in enumerate(self.param_names):
            metrics['rmse_per_param'][param] = np.sqrt(mean_squared_error(
                theta_true[:, i], theta_mean[:, i]
            ))
            metrics['mae_per_param'][param] = mean_absolute_error(
                theta_true[:, i], theta_mean[:, i]
            )
            metrics['bias_per_param'][param] = np.mean(theta_mean[:, i] - theta_true[:, i])
            
            # R² (correlation between true and estimated)
            correlation = np.corrcoef(theta_true[:, i], theta_mean[:, i])[0, 1]
            metrics['r2_per_param'][param] = correlation ** 2
        
        # ============================================================
        # 2. UNCERTAINTY QUANTIFICATION METRICS
        # ============================================================
        print("  Computing uncertainty metrics...")
        
        # Posterior variance per parameter (averaged over test set)
        posterior_var = posterior_samples.var(axis=1)  # (N_test, n_params)
        metrics['posterior_variance'] = {}
        metrics['posterior_std'] = {}
        
        for i, param in enumerate(self.param_names):
            metrics['posterior_variance'][param] = posterior_var[:, i].mean()
            metrics['posterior_std'][param] = np.sqrt(metrics['posterior_variance'][param])
        
        # Credible interval width (95%)
        metrics['ci_width_95'] = {}
        for i, param in enumerate(self.param_names):
            lower = np.percentile(posterior_samples[:, :, i], 2.5, axis=1)
            upper = np.percentile(posterior_samples[:, :, i], 97.5, axis=1)
            metrics['ci_width_95'][param] = (upper - lower).mean()
        
        # ============================================================
        # 3. CALIBRATION METRICS (Most Important!)
        # ============================================================
        print("  Computing calibration metrics...")
        
        # Coverage: fraction of true params inside 95% CI
        metrics['coverage_95'] = {}
        metrics['coverage_68'] = {}
        
        for i, param in enumerate(self.param_names):
            # 95% CI
            lower_95 = np.percentile(posterior_samples[:, :, i], 2.5, axis=1)
            upper_95 = np.percentile(posterior_samples[:, :, i], 97.5, axis=1)
            inside_95 = (theta_true[:, i] >= lower_95) & (theta_true[:, i] <= upper_95)
            metrics['coverage_95'][param] = inside_95.mean()
            
            # 68% CI (1-sigma)
            lower_68 = np.percentile(posterior_samples[:, :, i], 16, axis=1)
            upper_68 = np.percentile(posterior_samples[:, :, i], 84, axis=1)
            inside_68 = (theta_true[:, i] >= lower_68) & (theta_true[:, i] <= upper_68)
            metrics['coverage_68'][param] = inside_68.mean()
        
        # Expected Calibration Error (ECE)
        confidence_levels = np.linspace(0.1, 0.9, 9)
        metrics['ece_per_param'] = {}
        
        for i, param in enumerate(self.param_names):
            empirical_coverage = []
            for conf in confidence_levels:
                alpha = 1 - conf
                lower = np.percentile(posterior_samples[:, :, i], 100*alpha/2, axis=1)
                upper = np.percentile(posterior_samples[:, :, i], 100*(1-alpha/2), axis=1)
                inside = (theta_true[:, i] >= lower) & (theta_true[:, i] <= upper)
                empirical_coverage.append(inside.mean())
            
            empirical_coverage = np.array(empirical_coverage)
            ece = np.abs(empirical_coverage - confidence_levels).mean()
            metrics['ece_per_param'][param] = ece
        
        # Overall ECE (average over parameters)
        metrics['ece_overall'] = np.mean(list(metrics['ece_per_param'].values()))
        
        # Simultaneous coverage (all parameters jointly in 95% CI)
        all_inside = np.ones(N_test, dtype=bool)
        for i in range(self.n_params):
            lower = np.percentile(posterior_samples[:, :, i], 2.5, axis=1)
            upper = np.percentile(posterior_samples[:, :, i], 97.5, axis=1)
            all_inside &= (theta_true[:, i] >= lower) & (theta_true[:, i] <= upper)
        metrics['coverage_95_simultaneous'] = all_inside.mean()
        
        # ============================================================
        # 4. POSTERIOR STRUCTURE METRICS
        # ============================================================
        print("  Computing posterior structure metrics...")
        
        # Correlation matrix (averaged over test set)
        correlations = []
        for k in range(N_test):
            corr = np.corrcoef(posterior_samples[k].T)
            correlations.append(corr)
        
        avg_correlation = np.mean(correlations, axis=0)
        metrics['correlation_matrix'] = {}
        for i, param_i in enumerate(self.param_names):
            for j, param_j in enumerate(self.param_names):
                if i < j:  # Only upper triangle
                    key = f"{param_i}_{param_j}"
                    metrics['correlation_matrix'][key] = avg_correlation[i, j]
        
        # Covariance matrix (averaged)
        covariances = []
        for k in range(N_test):
            cov = np.cov(posterior_samples[k].T)
            covariances.append(cov)
        
        avg_covariance = np.mean(covariances, axis=0)
        metrics['covariance_matrix'] = avg_covariance
        
        # Condition number (identifiability diagnostic)
        condition_numbers = [np.linalg.cond(cov) for cov in covariances]
        metrics['condition_number_mean'] = np.mean(condition_numbers)
        metrics['condition_number_std'] = np.std(condition_numbers)
        
        # ============================================================
        # 5. COMPARISON METRICS (if prior provided)
        # ============================================================
        if prior_samples is not None:
            print("  Computing information gain metrics...")
            
            prior_var = prior_samples.var(axis=0)  # (n_params,)
            
            metrics['information_gain'] = {}
            metrics['variance_reduction_ratio'] = {}
            
            for i, param in enumerate(self.param_names):
                post_var = metrics['posterior_variance'][param]
                metrics['information_gain'][param] = prior_var[i] / post_var
                metrics['variance_reduction_ratio'][param] = 1 - (post_var / prior_var[i])
        
        # ============================================================
        # 6. LIKELIHOOD METRICS (if log_probs provided)
        # ============================================================
        if log_probs is not None:
            print("  Computing likelihood metrics...")
            
            metrics['mean_log_prob'] = log_probs.mean()
            metrics['median_log_prob'] = np.median(log_probs)
            metrics['std_log_prob'] = log_probs.std()
            metrics['nll'] = -log_probs.mean()  # Negative log-likelihood
            metrics['perplexity'] = np.exp(-log_probs.mean())
            
            # Fraction of "good" posteriors (log_prob > threshold)
            # A reasonable threshold might be based on the distribution
            threshold = np.percentile(log_probs, 10)  # Bottom 10%
            metrics['fraction_good_posteriors'] = (log_probs > threshold).mean()
        
        # ============================================================
        # 7. Z-SCORES (standardized errors)
        # ============================================================
        print("  Computing z-scores...")
        
        posterior_std = posterior_samples.std(axis=1)  # (N_test, n_params)
        z_scores = (theta_mean - theta_true) / posterior_std
        
        metrics['z_score_mean'] = {}
        metrics['z_score_std'] = {}
        
        for i, param in enumerate(self.param_names):
            metrics['z_score_mean'][param] = z_scores[:, i].mean()
            metrics['z_score_std'][param] = z_scores[:, i].std()
        
        # Ideally, z-scores should have mean ≈ 0 and std ≈ 1
        
        print("✓ All metrics computed!")
        
        return metrics
    
    def print_metrics_summary(self, metrics: Dict, verbose: bool = True):
        """
        Print a nicely formatted summary of metrics
        """
        print("\n" + "="*80)
        print("POSTERIOR EVALUATION METRICS SUMMARY")
        print("="*80)
        
        # Point estimation
        print("\n1. POINT ESTIMATION ACCURACY")
        print("-" * 80)
        print(f"  Overall RMSE: {metrics['rmse_overall']:.4f}")
        print(f"  Overall MAE:  {metrics['mae_overall']:.4f}")
        print("\n  Per-parameter:")
        for param in self.param_names:
            print(f"    {param:8s}: RMSE={metrics['rmse_per_param'][param]:.4f}, "
                  f"MAE={metrics['mae_per_param'][param]:.4f}, "
                  f"Bias={metrics['bias_per_param'][param]:+.4f}, "
                  f"R²={metrics['r2_per_param'][param]:.4f}")
        
        # Uncertainty
        print("\n2. UNCERTAINTY QUANTIFICATION")
        print("-" * 80)
        for param in self.param_names:
            print(f"  {param:8s}: Var={metrics['posterior_variance'][param]:.4f}, "
                  f"Std={metrics['posterior_std'][param]:.4f}, "
                  f"95% CI width={metrics['ci_width_95'][param]:.4f}")
        
        # Calibration (MOST IMPORTANT!)
        print("\n3. CALIBRATION (★ CRITICAL ★)")
        print("-" * 80)
        print(f"  Expected Calibration Error (ECE): {metrics['ece_overall']:.4f}")
        print(f"  Simultaneous 95% coverage: {metrics['coverage_95_simultaneous']:.2%}")
        print("\n  Per-parameter coverage:")
        for param in self.param_names:
            cov_95 = metrics['coverage_95'][param]
            cov_68 = metrics['coverage_68'][param]
            ece = metrics['ece_per_param'][param]
            
            # Check if well-calibrated
            cal_95 = "✓" if 0.93 <= cov_95 <= 0.97 else "✗"
            cal_68 = "✓" if 0.65 <= cov_68 <= 0.71 else "✗"
            
            print(f"    {param:8s}: 95% CI: {cov_95:.2%} {cal_95}  |  "
                  f"68% CI: {cov_68:.2%} {cal_68}  |  ECE: {ece:.4f}")
        
        # Correlation structure
        print("\n4. POSTERIOR CORRELATIONS")
        print("-" * 80)
        for key, val in metrics['correlation_matrix'].items():
            print(f"  ρ({key}): {val:+.3f}")
        print(f"\n  Condition number: {metrics['condition_number_mean']:.2f} "
              f"± {metrics['condition_number_std']:.2f}")
        
        # Information gain (if available)
        if 'information_gain' in metrics:
            print("\n5. INFORMATION GAIN (vs Prior)")
            print("-" * 80)
            for param in self.param_names:
                ig = metrics['information_gain'][param]
                vr = metrics['variance_reduction_ratio'][param]
                print(f"  {param:8s}: {ig:.2f}× reduction  "
                      f"({vr:.1%} variance reduction)")
        
        # Log probability (if available)
        if 'mean_log_prob' in metrics:
            print("\n6. LIKELIHOOD QUALITY")
            print("-" * 80)
            print(f"  Mean log prob: {metrics['mean_log_prob']:.2f}")
            print(f"  NLL: {metrics['nll']:.2f}")
            print(f"  Fraction good posteriors: {metrics['fraction_good_posteriors']:.1%}")
        
        # Z-scores
        print("\n7. Z-SCORE DIAGNOSTICS (should be ~N(0,1))")
        print("-" * 80)
        for param in self.param_names:
            z_mean = metrics['z_score_mean'][param]
            z_std = metrics['z_score_std'][param]
            
            # Check if reasonable
            mean_ok = "✓" if abs(z_mean) < 0.2 else "✗"
            std_ok = "✓" if 0.9 <= z_std <= 1.1 else "✗"
            
            print(f"  {param:8s}: mean={z_mean:+.3f} {mean_ok}  |  "
                  f"std={z_std:.3f} {std_ok}")
        
        print("\n" + "="*80 + "\n")
    
    def save_metrics_to_csv(self, metrics: Dict, filename: str):
        """
        Save metrics to CSV for easy comparison across experiments
        """
        rows = []
        
        # Flatten nested dictionaries
        for param in self.param_names:
            row = {
                'parameter': param,
                'rmse': metrics['rmse_per_param'][param],
                'mae': metrics['mae_per_param'][param],
                'bias': metrics['bias_per_param'][param],
                'r2': metrics['r2_per_param'][param],
                'posterior_variance': metrics['posterior_variance'][param],
                'posterior_std': metrics['posterior_std'][param],
                'ci_width_95': metrics['ci_width_95'][param],
                'coverage_95': metrics['coverage_95'][param],
                'coverage_68': metrics['coverage_68'][param],
                'ece': metrics['ece_per_param'][param],
                'z_score_mean': metrics['z_score_mean'][param],
                'z_score_std': metrics['z_score_std'][param],
            }
            
            if 'information_gain' in metrics:
                row['information_gain'] = metrics['information_gain'][param]
                row['variance_reduction'] = metrics['variance_reduction_ratio'][param]
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"✓ Saved metrics to {filename}")
        
        return df


# ============================================================
# USAGE EXAMPLE
# ============================================================





