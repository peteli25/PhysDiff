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

    def langevin_sample(self, x, energy_fn, n_steps=10, step_size=0.01):
        """Perform Langevin dynamics sampling to adjust generation path"""
        x_curr = x.clone()
        
        for _ in range(n_steps):
            x_curr.requires_grad_(True)
            energy = energy_fn(x_curr)
            grad = torch.autograd.grad(energy.sum(), x_curr)[0]
            x_curr.requires_grad_(False)
            
            noise = torch.randn_like(x_curr)
            x_curr = x_curr - step_size * grad + torch.sqrt(torch.tensor(2.0 * step_size, device=x_curr.device)) * noise
            
        return x_curr

    def q_sample(self, x_start, trend, batch_t, noise):
        """Apply forward diffusion with SWaT dataset-specific dimension handling"""
        batch_size, seq_len, feature_dim = x_start.shape
        
        if trend.ndim == 3:
            if trend.shape[1] != feature_dim:
                if trend.shape[1] == 1:
                    trend = trend.expand(-1, feature_dim, -1)
                else:
                    if trend.shape[2] == feature_dim:
                        trend = trend.transpose(1, 2)
                    else:
                        print(f"Reshaping trend from {trend.shape} to match {x_start.shape}")
                        new_trend = torch.zeros(batch_size, feature_dim, seq_len, device=trend.device)
                        
                        min_feat = min(trend.shape[1], feature_dim)
                        min_seq = min(trend.shape[2], seq_len)
                        
                        for b in range(batch_size):
                            new_trend[b, :min_feat, :min_seq] = trend[b, :min_feat, :min_seq]
                        
                        trend = new_trend
        
        if trend.shape[2] != seq_len:
            print(f"Adjusting trend sequence dimension from {trend.shape[2]} to {seq_len}")
            trend = torch.nn.functional.interpolate(
                trend, size=seq_len, mode='linear'
            )
        
        trend_reshaped = trend.transpose(1, 2) if trend.shape[1] != seq_len else trend
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, batch_t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, batch_t, x_start.shape)
        
        trend_influence = torch.sigmoid(10 * (1 - sqrt_alphas_cumprod_t))
        
        try:
            x_noisy = sqrt_alphas_cumprod_t * x_start + \
                    sqrt_one_minus_alphas_cumprod_t * (
                        (1 - trend_influence) * noise + 
                        trend_influence * trend_reshaped
                    )
        except RuntimeError as e:
            print(f"Diffusion calculation failed: {e}")
            print(f"Falling back to simple diffusion without trend")
            x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

        return x_noisy