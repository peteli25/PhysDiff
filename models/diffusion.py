import torch


class Diffusion:
    def __init__(self, time_steps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.betas = torch.linspace(beta_start, beta_end, time_steps).float().to(device)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.one_minus_sqrt_alphas_cumprod = 1. - torch.sqrt(self.alphas_cumprod)

    @staticmethod
    def _extract(data, batch_t, shape):
        batch_size = batch_t.shape[0]
        out = torch.gather(data, -1, batch_t)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))

    def q_sample(self, x_start, trend, batch_t, noise):
        """
        Apply the forward diffusion process
        """
        batch_size, seq_len, feature_dim = x_start.shape
        
        # Standardize trend tensor format
        if trend.ndim == 2:  # [channels, timesteps]
            trend = trend.unsqueeze(0).expand(batch_size, -1, -1)
        
        if trend.ndim == 3:
            # Adjust batch dimension if needed
            if trend.shape[0] != batch_size:
                trend = trend[:1].expand(batch_size, -1, -1)
            
            # Adjust feature dimension if needed
            if trend.shape[1] != feature_dim:
                if trend.shape[1] == 1:
                    trend = trend.expand(-1, feature_dim, -1)
                else:
                    trend = trend[:, :1].expand(-1, feature_dim, -1)
            
            # Adjust sequence length if needed
            if trend.shape[2] != seq_len:
                trend = torch.nn.functional.interpolate(
                    trend, size=seq_len, mode='linear'
                )
                
        # Reshape to match data format if needed
        if trend.shape[1] == seq_len and trend.shape[2] == feature_dim:
            trend_reshaped = trend
        else:
            trend_reshaped = trend.transpose(1, 2)
        
        # Verify dimensions
        assert trend_reshaped.shape[0] == batch_size, f"Batch size mismatch: trend_reshaped {trend_reshaped.shape[0]} vs required {batch_size}"
        assert trend_reshaped.shape[1] == seq_len, f"Sequence length mismatch: trend_reshaped {trend_reshaped.shape[1]} vs required {seq_len}"
        assert trend_reshaped.shape[2] == feature_dim, f"Feature dimension mismatch: trend_reshaped {trend_reshaped.shape[2]} vs required {feature_dim}"
        
        # Extract diffusion coefficients
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, batch_t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, batch_t, x_start.shape)
        
        # Dynamic trend influence - stronger at early steps, weaker at later steps
        trend_influence = torch.sigmoid(10 * (1 - sqrt_alphas_cumprod_t))
        
        # Apply enhanced diffusion with dynamic trend influence
        x_noisy = sqrt_alphas_cumprod_t * x_start + \
                sqrt_one_minus_alphas_cumprod_t * (
                    (1 - trend_influence) * noise + 
                    trend_influence * trend_reshaped
                )

        return x_noisy