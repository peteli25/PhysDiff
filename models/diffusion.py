import torch

class Diffusion:
    def __init__(self, time_steps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.device = torch.device(device)
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

    def q_sample(self, x_start, trend, batch_t, noise, high_freq=None):
        batch_size, seq_len, feature_dim = x_start.shape
        
        # Standardize trend tensor format
        if trend.ndim == 2:
            trend = trend.unsqueeze(0).expand(batch_size, -1, -1)
        
        if trend.ndim == 3:
            if trend.shape[0] != batch_size:
                trend = trend[:1].expand(batch_size, -1, -1)
            
            if trend.shape[1] != feature_dim:
                if trend.shape[1] == 1:
                    trend = trend.expand(-1, feature_dim, -1)
                else:
                    trend = trend[:, :1].expand(-1, feature_dim, -1)
            
            if trend.shape[2] != seq_len:
                trend = torch.nn.functional.interpolate(
                    trend, size=seq_len, mode='linear'
                )
                
        # Reshape to match data format if needed
        if trend.shape[1] == seq_len and trend.shape[2] == feature_dim:
            trend_reshaped = trend
        else:
            trend_reshaped = trend.transpose(1, 2)
        
        # Process high frequency tensor if provided
        if high_freq is not None:
            if high_freq.ndim == 2:
                high_freq = high_freq.unsqueeze(0).expand(batch_size, -1, -1)
            
            if high_freq.ndim == 3:
                if high_freq.shape[0] != batch_size:
                    high_freq = high_freq[:1].expand(batch_size, -1, -1)
                
                if high_freq.shape[1] != feature_dim:
                    if high_freq.shape[1] == 1:
                        high_freq = high_freq.expand(-1, feature_dim, -1)
                    else:
                        high_freq = high_freq[:, :1].expand(-1, feature_dim, -1)
                
                if high_freq.shape[2] != seq_len:
                    high_freq = torch.nn.functional.interpolate(
                        high_freq, size=seq_len, mode='linear'
                    )
                    
            # Reshape high_freq to match data format if needed
            if high_freq.shape[1] == seq_len and high_freq.shape[2] == feature_dim:
                high_freq_reshaped = high_freq
            else:
                high_freq_reshaped = high_freq.transpose(1, 2)
        else:
            high_freq_reshaped = torch.zeros_like(trend_reshaped)
        
        # Extract diffusion coefficients
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, batch_t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, batch_t, x_start.shape)
        
        # Dynamic frequency-based diffusion
        diffusion_progress = sqrt_one_minus_alphas_cumprod_t
        
        # Frequency-adaptive influence factors
        low_freq_ratio = torch.sigmoid(5 * (1 - diffusion_progress))
        high_freq_ratio = torch.sigmoid(5 * diffusion_progress)
        
        # Balance factor to weight noise vs frequency guided info
        noise_weight = 0.7
        
        # Enhanced diffusion with frequency guided information
        x_noisy = sqrt_alphas_cumprod_t * x_start + \
                  sqrt_one_minus_alphas_cumprod_t * (
                      noise_weight * noise + 
                      (1 - noise_weight) * (low_freq_ratio * trend_reshaped + high_freq_ratio * high_freq_reshaped)
                  )

        return x_noisy