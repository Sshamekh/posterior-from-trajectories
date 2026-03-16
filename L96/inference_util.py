import torch
import numpy as np
def evaluate_conditional_nf(model, data_loader, device="cpu", n_samples=10):
    """
    Evaluate ConditionalNF and return all generated samples along with true values.
    
    Args:
        model: trained ConditionalNF model
        data_loader: DataLoader yielding (c_batch, x_batch)
        device: "cuda" or "cpu"
        n_samples: number of samples to draw for each conditioning input
        
    Returns:
        avg_loss: Average NLL
        all_c: Conditioning variables (torch.Tensor)
        all_x_true: True parameters (torch.Tensor)
        all_x_samples: All generated samples (torch.Tensor, shape = (total_batches*n_samples, x_dim))
    """
    model.eval()
    total_loss = 0.0
    
    all_c = []
    all_x_true = []
    all_x_samples = []
    
    with torch.no_grad():
        for c_batch, x_batch in data_loader:
            c_batch, x_batch = c_batch.to(device), x_batch.to(device)

            # Compute loss
            loss = model(x_batch, c_batch)
            total_loss += loss.item() * x_batch.size(0)
            
            # Store true values
            all_c.append(c_batch.cpu())
            all_x_true.append(x_batch.cpu())
            
            # Generate all samples
            samples = model.sample(c_batch, n_samples=n_samples, device=device)  
            all_x_samples.append(samples.cpu())
    
    avg_loss = total_loss / len(data_loader.dataset)
    
    # Stack all results
    all_c = torch.cat(all_c, dim=0).numpy()
    all_x_true = torch.cat(all_x_true, dim=0).numpy()
    all_x_samples = torch.cat(all_x_samples, dim=0).numpy()  # shape: (n_samples, total_points, x_dim)
    
    print(f"Average NLL on evaluation set: {avg_loss:.4f}")
    return avg_loss, all_c, all_x_true, all_x_samples