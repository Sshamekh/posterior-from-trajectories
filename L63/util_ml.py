import torch
import os
import numpy as np

def configure_device(device_id=0):
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        device = torch.device(f'cuda:{device_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(device_id)} (ID: {device_id})")
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU.")
    return device

def load_from_checkpoint(model, optimizer, checkpoint_path, device):
    if os.path.exists(checkpoint_path):
        # 1) load both model & optimizer onto the correct device
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # 2) move optimizer's internal tensors (momentum buffers, etc.) to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print(f" Loaded checkpoint '{checkpoint_path}'")
    else:
        print(f" No checkpoint found at '{checkpoint_path}' — training from scratch.")
    return model

def adjust_lr(optimizer, decay_rate=0.98, min_lr=1e-5):
    for param_group in optimizer.param_groups:
        new_lr = param_group['lr'] * decay_rate
        param_group['lr'] = max(new_lr, min_lr)


class EarlyStop:
    """Used to early stop the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=False, delta=0, 
                 save_name="checkpoint.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            save_name (string): The filename with which the model and the optimizer is saved when improved.
                            Default: "checkpoint.pt"
        """
        self.patience = patience
        self.verbose = verbose
        self.save_name = save_name
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
            self.counter = 0
            
        return self.early_stop
    
    def save_checkpoint(self, val_loss, model, optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        state = {"net":model.state_dict(), "optimizer":optimizer.state_dict()}
        torch.save(state, self.save_name)
        self.val_loss_min = val_loss
