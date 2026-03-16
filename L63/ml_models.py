import torch
import torch.nn as nn
import torch.optim as optim
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms import CompositeTransform, ReversePermutation
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
import os
import torch.nn.functional as F

class ConditionalNF(nn.Module):
    def __init__(self, x_dim, cond_dim, n_time=1, hidden_dims=[64, 64], n_flow_layers=2):
        super().__init__()
        self.x_dim = x_dim
        self.cond_dim = cond_dim
        self.n_time = n_time

        # Effective conditioning dimension after flattening
        cond_input_dim = cond_dim * n_time

        # 1) Context encoder: maps flattened C → embedding for the flow conditioner
        cond_layers = []
        inp = cond_input_dim
        for h in hidden_dims:
            cond_layers += [nn.Linear(inp, h), nn.ReLU()]
            inp = h
        self.context_net = nn.Sequential(*cond_layers)

        # 2) Build a stack of conditional MADE transforms + permutations
        transforms = []
        for _ in range(n_flow_layers):
            transforms.append(ReversePermutation(features=x_dim))
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=x_dim,
                    hidden_features=hidden_dims[0],
                    context_features=hidden_dims[-1]
                )
            )
        flow_transform = CompositeTransform(transforms)
        self.flow_transform = flow_transform

        # 3) Base distribution
        base_dist = StandardNormal([x_dim])

        # 4) The conditional flow
        self.flow = Flow(self.flow_transform, base_dist, self.context_net)

        # Post‐flow activation to enforce non-negativity
        self.output_act = nn.ReLU()

    def _flatten_context(self, c):
        """Flatten context if it has time dimension."""
        if c.ndim == 3:  # (batch, time, features)
            c = c.reshape(c.size(0), -1)
        return c

    def forward(self, x, c):
        """Compute negative log-likelihood loss."""
        c = self._flatten_context(c)
        log_prob = self.flow.log_prob(inputs=x, context=c)
        return -log_prob.mean()

    @torch.no_grad()
    def sample(self, c, n_samples=100, device=None):
        """Draw samples x ∼ p(x|C)."""
        if device is None:
            device = next(self.parameters()).device
        c = c.to(device)
        c = self._flatten_context(c)
        samples = self.flow.sample(n_samples, context=c)
        return samples


class ConditionalNF_LSTM(nn.Module):
    def __init__(self, x_dim, cond_dim, n_time, hidden_dims=[64, 64], lstm_hidden=64, n_flow_layers=2):
        super().__init__()
        self.x_dim = x_dim
        self.cond_dim = cond_dim
        self.n_time = n_time
        self.lstm_hidden = lstm_hidden

        # 1) LSTM-based context encoder: (batch, time, features) → context embedding
        self.lstm = nn.LSTM(input_size=cond_dim,
                            hidden_size=lstm_hidden,
                            num_layers=1,
                            batch_first=True)

        # Feed LSTM output (final hidden state) into MLP to match flow context size
        cond_layers = []
        inp = lstm_hidden
        for h in hidden_dims:
            cond_layers += [nn.Linear(inp, h), nn.ReLU()]
            inp = h
        self.context_net = nn.Sequential(*cond_layers)

        # 2) Flow layers
        transforms = []
        for _ in range(n_flow_layers):
            transforms.append(ReversePermutation(features=x_dim))
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=x_dim,
                    hidden_features=hidden_dims[0],
                    context_features=hidden_dims[-1]
                )
            )
        flow_transform = CompositeTransform(transforms)
        self.flow_transform = flow_transform

        # 3) Base distribution
        base_dist = StandardNormal([x_dim])

        # 4) The conditional flow
        self.flow = Flow(self.flow_transform, base_dist, self.context_net)


    def _encode_context(self, c):
        """
        Encode time-series context with LSTM and MLP.
        Input: c (batch, time, features)
        Output: context embedding (batch, hidden_dims[-1])
        """
        if c.ndim == 2:
            c = c.unsqueeze(1)  # handle single-time-step case
        _, (h_n, _) = self.lstm(c)  # h_n: (num_layers, batch, hidden)
        c_embed = h_n[-1]           # take last layer hidden state (batch, lstm_hidden)
        return c_embed

    def forward(self, x, c):
        """Compute negative log-likelihood loss."""
        c_embed = self._encode_context(c)
        log_prob = self.flow.log_prob(inputs=x, context=c_embed)
        return -log_prob.mean()

    @torch.no_grad()
    def sample(self, c, n_samples=100, device=None):
        """Draw samples x ∼ p(x|C)."""
        if device is None:
            device = next(self.parameters()).device
        c = c.to(device)
        c_embed = self._encode_context(c)
        samples = self.flow.sample(n_samples, context=c_embed)
        return samples



class ConditionalNF_LSTM_ANN(nn.Module):
    def __init__(self, x_dim, cond_dim, n_time, hidden_dims=[128, 128], lstm_hidden=64, n_flow_layers=2):
        super().__init__()
        self.x_dim = x_dim
        self.cond_dim = cond_dim
        self.n_time = n_time
        self.lstm_hidden = lstm_hidden

        # 1) LSTM for time series encoding
        self.lstm = nn.LSTM(input_size=cond_dim,
                            hidden_size=lstm_hidden,
                            num_layers=1,
                            batch_first=True)

        # 2) Dense layers after LSTM output to form context vector
        dense_layers = []
        inp = lstm_hidden
        for h in hidden_dims:
            dense_layers += [nn.Linear(inp, h), nn.ReLU()]
            inp = h
        self.context_net = nn.Sequential(*dense_layers)

        # 3) Flow layers
        transforms = []
        for _ in range(n_flow_layers):
            transforms.append(ReversePermutation(features=x_dim))
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=x_dim,
                    hidden_features=hidden_dims[0],
                    context_features=hidden_dims[-1]
                )
            )
        flow_transform = CompositeTransform(transforms)

        # Base distribution
        base_dist = StandardNormal([x_dim])

        # Flow
        self.flow = Flow(flow_transform, base_dist, self.context_net)

    def _encode_context(self, c):
        """
        Encode context: pass through LSTM, take full sequence output,
        pool over time, then pass to dense layers.
        """
        if c.ndim == 2:
            c = c.unsqueeze(1)  # Handle single timestep case
        
        output, _ = self.lstm(c)   # output shape: (batch, time, lstm_hidden)
        
        # Mean pooling over time
        c_pooled = output.mean(dim=1)  # shape: (batch, lstm_hidden)
        
        # Pass pooled features to ANN
        c_embed = self.context_net(c_pooled)
        return c_embed

    def forward(self, x, c):
        """Compute negative log-likelihood loss."""
        c_embed = self._encode_context(c)
        log_prob = self.flow.log_prob(inputs=x, context=c_embed)
        return -log_prob.mean()

    @torch.no_grad()
    def sample(self, c, n_samples=100, device=None):
        """Draw samples x ∼ p(x|C)."""
        if device is None:
            device = next(self.parameters()).device
        c = c.to(device)
        c_embed = self._encode_context(c)
        samples = self.flow.sample(n_samples, context=c_embed)
        return samples




def train_conditional_nf(
    model,
    train_loader,
    val_loader,
    adjust_lr,
    early_stop,
    optimizer,
    epochs=100,
    device="cpu"
):
    """
    Train ConditionalNF for parameter estimation with early stopping.

    Args:
        model: ConditionalNF
        train_loader: DataLoader yielding (x_batch, c_batch)
        val_loader: DataLoader yielding (x_batch, c_batch)
        adjust_lr: function to adjust learning rate
        early_stop: function for early stopping
        optimizer: torch optimizer
        epochs: max number of epochs
        device: "cuda" or "cpu"
    """

    model = model.to(device)

    for epoch in range(1, epochs + 1):
        # ----- Training -----
        model.train()
        running_train_loss = 0.0
        for c_batch, x_batch in train_loader:
            x_batch, c_batch = x_batch.to(device), c_batch.to(device)

            optimizer.zero_grad()
            loss = model(x_batch, c_batch)  # NLL returned directly
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * x_batch.size(0)

        avg_train_loss = running_train_loss / len(train_loader.dataset)
        adjust_lr(optimizer)

        # ----- Validation -----
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for c_batch, x_batch in val_loader:
                x_batch, c_batch = x_batch.to(device), c_batch.to(device)
                loss = model(x_batch, c_batch)
                running_val_loss += loss.item() * x_batch.size(0)

        avg_val_loss = running_val_loss / len(val_loader.dataset)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train NLL: {avg_train_loss:.4f} | "
            f"Val NLL:   {avg_val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # ---------- Early stopping ----------
        if early_stop(avg_val_loss, model, optimizer):
            break
class ConditionalNF_LSTM_v2(nn.Module):# (self, x_dim, cond_dim, n_time, hidden_dims=[128, 128], lstm_hidden=64, n_flow_layers=2)
    def __init__(self, x_dim, cond_dim, n_time, hidden_dims=[64, 64], 
                 lstm_hidden=64, lstm_layers=1, lstm_dropout=0.2,
                 context_size=64, n_flow_layers=4, bidirectional=False):
        """
        Conditional Normalizing Flow with LSTM encoder.
        
        Args:
            x_dim: Dimension of the target variable (parameters to model)
            cond_dim: Dimension of input features at each time step
            hidden_dims: Hidden dimensions for the flow transforms
            lstm_hidden: Hidden size of LSTM
            lstm_layers: Number of LSTM layers (≥1)
            lstm_dropout: Dropout rate for LSTM (only applies if lstm_layers > 1)
            context_size: Size of the context vector passed to the flow
            n_flow_layers: Number of flow transformation layers
            bidirectional: If True, use bidirectional LSTM
        """
        super().__init__()
        self.x_dim = x_dim
        self.cond_dim = cond_dim
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.context_size = context_size

        # 1) LSTM encoder
        self.lstm = nn.LSTM(
            input_size=cond_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # Calculate LSTM output dimension
        lstm_output_dim = lstm_hidden * (2 if bidirectional else 1)

        # 2) Projection layer to match context_size (if needed)
        if lstm_output_dim != context_size:
            self.context_projection = nn.Linear(lstm_output_dim, context_size)
        else:
            self.context_projection = nn.Identity()

        # 3) Flow layers
        transforms = []
        for _ in range(n_flow_layers):
            transforms.append(ReversePermutation(features=x_dim))
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=x_dim,
                    hidden_features=hidden_dims[0],
                    context_features=context_size  # Use context_size here
                )
            )
        flow_transform = CompositeTransform(transforms)

        # 4) Base distribution
        base_dist = StandardNormal([x_dim])

        # 5) The conditional flow (context_net is now just the projection)
        self.flow = Flow(flow_transform, base_dist, self.context_projection)

    def _encode_context(self, c):
        """
        Encode time-series context with LSTM.
        
        Args:
            c: (batch, time, features) - temporal trajectories
        
        Returns:
            context embedding: (batch, context_size)
        """
        # Handle 2D input (single time step)
        if c.ndim == 2:
            c = c.unsqueeze(1)  # (batch, features) -> (batch, 1, features)
        
        # Process through LSTM
        # output: (batch, time, lstm_hidden * num_directions)
        # h_n: (num_layers * num_directions, batch, lstm_hidden)
        _, (h_n, _) = self.lstm(c)
        
        # Extract final hidden state(s)
        if self.bidirectional:
            # Concatenate forward and backward final states
            # Forward: h_n[-2], Backward: h_n[-1]
            c_embed = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, lstm_hidden*2)
        else:
            # Take the last layer's hidden state
            c_embed = h_n[-1]  # (batch, lstm_hidden)
        
        return c_embed

    def forward(self, x, c):
        """
        Compute negative log-likelihood loss.
        
        Args:
            x: (batch, x_dim) - target variables (parameters)
            c: (batch, time, cond_dim) - conditioning trajectories
        
        Returns:
            Scalar loss (negative log-likelihood)
        """
        c_embed = self._encode_context(c)
        log_prob = self.flow.log_prob(inputs=x, context=c_embed)
        return -log_prob.mean()

    @torch.no_grad()
    def sample(self, c, n_samples=100, device=None):
        """
        Draw samples x ∼ p(x|C).
        
        Args:
            c: (batch, time, cond_dim) or (time, cond_dim) - conditioning trajectories
            n_samples: Number of samples to draw
            device: Device to use for sampling
        
        Returns:
            samples: (n_samples, batch, x_dim) or (n_samples, x_dim)
        """
        if device is None:
            device = next(self.parameters()).device
        c = c.to(device)
        
        # Handle single trajectory input
        if c.ndim == 2:
            c = c.unsqueeze(0)  # (time, features) -> (1, time, features)
        
        c_embed = self._encode_context(c)
        samples = self.flow.sample(n_samples, context=c_embed)
        return samples