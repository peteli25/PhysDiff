import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from models.mafd import MAFD
from models.aspe import ASPE, MultiScaleASPE
from models.attention import OrdAttention, MixAttention
from models.block import SpatialTemporalTransformerBlock, DecompositionBlock
from models.diffusion import Diffusion
from models.embedding import DataEmbedding, PositionEmbedding, TimeEmbedding


class PhysicalFeatureExtractor(nn.Module):
    def __init__(self, n_components=8, dic_dist=0.05, max_mag=0.95, device=None, aspe_m_range=[2, 3, 4], aspe_tau_range=[1, 2, 3]):
        super(PhysicalFeatureExtractor, self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        
        self.mafd = MAFD(device=self.device)
        self.n_components = n_components
        self.dic_dist = dic_dist
        self.max_mag = max_mag
        
        self.aspe = MultiScaleASPE(m_range=aspe_m_range, tau_range=aspe_tau_range, device=self.device)
    
    def _prepare_signal(self, x):
        """Prepare signal for MAFD"""
        if len(x.shape) == 3:
            x = x[0].transpose(0, 1)
        return x
    
    def _extract_components(self, mafd_results):
        """Extract high and low frequency components from MAFD results"""
        high_freq = []
        low_freq = []
        
        for ch_idx in range(len(mafd_results['components'])):
            components = mafd_results['components'][ch_idx]
            
            if len(components) == 0:
                continue

            mid_idx = len(components) // 2
            
            high_freq_components = []
            for comp in components[:mid_idx]:
                comp_cpu = comp.cpu() if hasattr(comp, 'cpu') else comp
                if np.iscomplexobj(comp_cpu):
                    comp_cpu = np.real(comp_cpu)
                high_freq_components.append(comp_cpu)
                
            low_freq_components = []
            for comp in components[mid_idx:]:
                comp_cpu = comp.cpu() if hasattr(comp, 'cpu') else comp
                if np.iscomplexobj(comp_cpu):
                    comp_cpu = np.real(comp_cpu)
                low_freq_components.append(comp_cpu)
                
            if high_freq_components:
                high_freq.append(np.sum(high_freq_components, axis=0))
            if low_freq_components:
                low_freq.append(np.sum(low_freq_components, axis=0))
        
        if high_freq:
            high_freq = torch.tensor(np.array(high_freq), dtype=torch.float32, device=self.device)
        else:
            high_freq = torch.zeros((1, mafd_results['original'].shape[1]), dtype=torch.float32, device=self.device)
            
        if low_freq:
            low_freq = torch.tensor(np.array(low_freq), dtype=torch.float32, device=self.device)
        else:
            low_freq = torch.zeros((1, mafd_results['original'].shape[1]), dtype=torch.float32, device=self.device)
            
        return high_freq, low_freq
    
    def compute_aspe_weights(self, signal, window_size=64, stride=1):
        """Compute ASPE weights for time series segments"""
        if len(signal.shape) == 3:
            signal = signal[0]
        
        if len(signal.shape) == 2:
            signal = signal[0]
            
        signal_np = signal.cpu().numpy() if isinstance(signal, torch.Tensor) else signal
            
        weights = []
        for i in range(0, len(signal_np) - window_size + 1, stride):
            segment = signal_np[i:i+window_size]
            aspe_val = self.aspe.compute_mean(segment)
            weights.append(aspe_val)
            
        if len(weights) > 0:
            weights = np.array(weights)
            min_val, max_val = weights.min(), weights.max()
            if max_val > min_val:
                weights = (weights - min_val) / (max_val - min_val)
            else:
                weights = np.ones_like(weights)
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            weights = torch.ones(1, dtype=torch.float32, device=self.device)
            
        return weights
    
    def compute_time_freq_distribution(self, signal):
        """Compute time-frequency distribution for KL divergence"""
        from scipy import signal as sp_signal
        
        signal_np = signal.cpu().numpy() if isinstance(signal, torch.Tensor) else signal
        
        freqs, times, Sxx = sp_signal.spectrogram(signal_np, fs=1.0, nperseg=min(256, len(signal_np)))
        
        Sxx = Sxx / (Sxx.sum() + 1e-10)
        
        return torch.tensor(Sxx, dtype=torch.float32, device=self.device)
    
    def forward(self, x, compute_tfd=False, batch_processing=True):
        """Extract physical features with improved batch handling"""
        batch_size = x.shape[0]
        
        if batch_processing:
            batch_indices = range(batch_size)
        else:
            if batch_size <= 4:
                batch_indices = range(batch_size)
            else:
                batch_indices = np.linspace(0, batch_size-1, 4, dtype=int)
        
        all_high_freq = []
        all_low_freq = []
        all_complexity = []
        all_tfd = [] if compute_tfd else None
        
        for b in batch_indices:
            try:
                x_signal = self._prepare_signal(x[b:b+1])
                
                aspe_weights = self.compute_aspe_weights(x_signal)
                
                self.mafd.loadInputSignal(x_signal)
                self.mafd.genDic(dist=self.dic_dist, max_an_mag=self.max_mag)
                self.mafd.genEva()
                self.mafd.init_decomp()
                
                for _ in range(self.n_components - 1):
                    self.mafd.nextDecomp(aspe_weights=aspe_weights)
                
                orig_data = self.mafd.s.cpu().numpy()
                recon = self.mafd.reconstruct(self.n_components - 1).cpu().numpy()
                
                mafd_results = {
                    'original': orig_data,
                    'reconstructed': recon,
                    'components': [self.mafd.deComp[ch] for ch in range(len(self.mafd.deComp))],
                    'parameters': {
                        'a': [self.mafd.an[ch] for ch in range(len(self.mafd.an))],
                        'coef': [self.mafd.coef[ch] for ch in range(len(self.mafd.coef))],
                        'level': self.n_components
                    }
                }
                
                high_freq, low_freq = self._extract_components(mafd_results)
                first_channel = orig_data[0] if orig_data.ndim > 1 and orig_data.shape[0] > 0 else orig_data
                complexity = self.aspe.compute_mean(first_channel)
                
                all_high_freq.append(high_freq)
                all_low_freq.append(low_freq)
                all_complexity.append(complexity)
                
                if compute_tfd:
                    tfd = self.compute_time_freq_distribution(first_channel)
                    all_tfd.append(tfd)
                    
            except Exception as e:
                print(f"Feature extraction failed for batch {b}: {str(e)}")
                seq_len = x[b:b+1].shape[1]
                all_high_freq.append(torch.zeros((1, seq_len), dtype=torch.float32, device=self.device))
                all_low_freq.append(torch.zeros((1, seq_len), dtype=torch.float32, device=self.device))
                all_complexity.append(0.5)
                
                if compute_tfd:
                    all_tfd.append(torch.ones((10, 10), dtype=torch.float32, device=self.device) / 100.0)
        
        if batch_processing:
            try:
                stacked_high_freq = torch.stack(all_high_freq, dim=0)
                stacked_low_freq = torch.stack(all_low_freq, dim=0)
            except:
                # Create consistent tensors
                seq_len = x.shape[1]
                stacked_high_freq = torch.zeros((batch_size, 1, seq_len), dtype=torch.float32, device=self.device)
                stacked_low_freq = torch.zeros((batch_size, 1, seq_len), dtype=torch.float32, device=self.device)
                
                # Fill in available data
                for i, (hf, lf) in enumerate(zip(all_high_freq, all_low_freq)):
                    if i < batch_size:
                        if hf.shape[1] == seq_len:
                            stacked_high_freq[i, :, :] = hf
                        if lf.shape[1] == seq_len:
                            stacked_low_freq[i, :, :] = lf
                
            avg_complexity = sum(all_complexity) / len(all_complexity)
            
            if compute_tfd:
                avg_tfd = torch.stack(all_tfd, dim=0).mean(dim=0)
                return stacked_high_freq, stacked_low_freq, avg_complexity, avg_tfd
            else:
                return stacked_high_freq, stacked_low_freq, avg_complexity
        else:
            if compute_tfd and all_tfd:
                return all_high_freq[0], all_low_freq[0], all_complexity[0], all_tfd[0]
            else:
                return all_high_freq[0], all_low_freq[0], all_complexity[0]


class RoutingAttention(nn.Module):
    def __init__(self, model_dim, atten_dim, head_num, dropout, residual=True):
        super(RoutingAttention, self).__init__()
        self.atten_dim = atten_dim
        self.head_num = head_num
        self.residual = residual
        self.model_dim = model_dim
        
        self.W_Q = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_K = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_V = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_Ph_high = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_Ph_low = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        
        self.fc = nn.Linear(self.atten_dim * self.head_num, model_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)
        
    def _ensure_tensor_shape(self, tensor, target_shape, name="tensor"):
        """Ensure tensor has the correct shape with improved dimensionality handling"""
        batch_size, seq_len, model_dim = target_shape
        
        # Check dimensionality and reshape if needed
        if tensor.dim() == 4:  # Handle 4D tensors like [64, 64, 1, 512]
            tensor = tensor.squeeze(2)  # Remove the third dimension
        elif tensor.dim() > 4:
            tensor = tensor.view(tensor.size(0), tensor.size(1), -1)
            
        # Handle size mismatch
        if tensor.size(0) != batch_size or tensor.size(1) != seq_len or tensor.size(2) != model_dim:
            result = torch.zeros(target_shape, device=tensor.device, dtype=tensor.dtype)
            
            # Set proper copy dimensions
            src_batch = min(tensor.size(0), batch_size)
            src_seq = min(tensor.size(1), seq_len)
            src_dim = min(tensor.size(2), model_dim)
            
            # Copy only valid dimensions
            result[:src_batch, :src_seq, :src_dim] = tensor[:src_batch, :src_seq, :src_dim]
            return result
        
        return tensor
        
    def forward(self, Q, K, V, Ph_high, Ph_low):
        """Forward pass with improved tensor handling"""
        residual = Q.clone()
        batch_size, seq_len = Q.size(0), Q.size(1)
        target_shape = (batch_size, seq_len, self.model_dim)
        
        # Ensure physical features have the correct shape
        Ph_high = self._ensure_tensor_shape(Ph_high, target_shape, "Ph_high")
        Ph_low = self._ensure_tensor_shape(Ph_low, target_shape, "Ph_low")
        
        # Project tensors
        Q = self.W_Q(Q).view(batch_size, seq_len, self.head_num, self.atten_dim)
        K = self.W_K(K).view(batch_size, seq_len, self.head_num, self.atten_dim)
        V = self.W_V(V).view(batch_size, seq_len, self.head_num, self.atten_dim)
        Ph_high = self.W_Ph_high(Ph_high).view(batch_size, seq_len, self.head_num, self.atten_dim)
        Ph_low = self.W_Ph_low(Ph_low).view(batch_size, seq_len, self.head_num, self.atten_dim)
        
        # Transpose for attention calculation
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        Ph_high = Ph_high.permute(0, 2, 1, 3)
        Ph_low = Ph_low.permute(0, 2, 1, 3)
        
        # Calculate attention scores
        scores_data = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        scores_high = torch.matmul(Q, Ph_high.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        scores_low = torch.matmul(Q, Ph_low.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        
        # Dynamic routing weights
        gate_high = torch.sigmoid(scores_high.mean(dim=-1, keepdim=True))
        gate_low = torch.sigmoid(scores_low.mean(dim=-1, keepdim=True))
        
        # Combined attention with adaptive weights
        attn = F.softmax(scores_data + gate_high * scores_high + gate_low * scores_low, dim=-1)
        context = torch.matmul(attn, V)
        
        # Format output
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        output = self.dropout(self.fc(context))
        
        return self.norm(output + residual) if self.residual else self.norm(output)


class PhysicalGuidedBlock(nn.Module):
    """
    Transformer block with physically-guided attention
    """
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, head_num, dropout):
        super(PhysicalGuidedBlock, self).__init__()
        self.routing_attention = RoutingAttention(model_dim, atten_dim, head_num, dropout)
        
        self.conv1 = nn.Conv1d(in_channels=model_dim, out_channels=ff_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=1)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)
        self.activation = F.gelu
        
    def forward(self, x, ph_high, ph_low):
        """
        Forward pass for physically-guided block
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
        ph_high : torch.Tensor
            High frequency physical features
        ph_low : torch.Tensor
            Low frequency physical features
            
        Returns:
        --------
        torch.Tensor: Output tensor
        """
        # Apply routing attention
        x = self.routing_attention(x, x, x, ph_high, ph_low)
        
        # Apply feed-forward network
        residual = x.clone()
        x = self.activation(self.conv1(x.transpose(1, 2)))
        x = self.dropout(self.conv2(x).transpose(1, 2))
        
        return self.norm(x + residual)


# detector.py - Improved PhysicallyGuidedDiffusion

class PhysicallyGuidedDiffusion(nn.Module):
    def __init__(self, time_steps=1000, beta_start=0.0001, beta_end=0.02, 
                 window_size=64, model_dim=512, ff_dim=2048, atten_dim=64,
                 feature_num=8, time_num=5, block_num=2, head_num=8,
                 dropout=0.1, device=None, d=30, t=500, mafd_components=8,
                 dataset='SWaT', aspe_m_range=[2, 3, 4], aspe_tau_range=[1, 2, 3],
                 use_langevin=True, langevin_steps=10, langevin_step_size=0.01):
        super(PhysicallyGuidedDiffusion, self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        
        # Model parameters
        self.window_size = window_size
        self.model_dim = model_dim
        self.feature_num = feature_num
        self.time_num = time_num
        self.t = t  # Noise step for diffusion
        self.dataset = dataset

        
        # Langevin dynamics parameters
        self.use_langevin = use_langevin
        self.langevin_steps = langevin_steps
        self.langevin_step_size = langevin_step_size

        if dataset in ['MSL', 'SMAP']:
            self.feature_extractor = PhysicalFeatureExtractor(
                n_components=min(4, mafd_components),
                device=self.device,
                aspe_m_range=[2, 3], 
                aspe_tau_range=[1] 
            )
        else:
            self.feature_extractor = PhysicalFeatureExtractor(
                n_components=mafd_components, 
                device=self.device,
                aspe_m_range=aspe_m_range,
                aspe_tau_range=aspe_tau_range
            )
        
        self.diffusion = Diffusion(
            time_steps=time_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=self.device
        )
        
        # Embedding layers
        self.data_embedding = DataEmbedding(model_dim, feature_num)
        self.time_embedding = TimeEmbedding(model_dim, time_num)
        self.position_embedding = PositionEmbedding(model_dim)
        
        # Feature projections
        self.high_freq_proj = nn.Linear(1, model_dim)
        self.low_freq_proj = nn.Linear(1, model_dim)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        for i in range(block_num):
            self.encoder_blocks.append(
                PhysicalGuidedBlock(
                    window_size=window_size,
                    model_dim=model_dim,
                    ff_dim=ff_dim,
                    atten_dim=atten_dim,
                    head_num=head_num,
                    dropout=dropout
                )
            )
            
        # Output heads
        self.reconstruction_head = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, feature_num)
        )
        
        self.anomaly_predictor = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 1)
        )
        
        # TFD analysis
        self.prior_tfd = None
    
    def compute_energy_function(self, x, trend):
        """Physical energy function for Langevin dynamics"""
        mse = torch.mean((x - trend) ** 2)
        
        if x.dim() > 2:  # [batch, seq, features]
            # Temporal smoothness constraint
            temp_grad = x[:, 1:, :] - x[:, :-1, :]
            smoothness = torch.mean(temp_grad ** 2)
            return mse + 0.1 * smoothness
        else:
            return mse
    
    def compute_kl_divergence(self, p, q):
        """KL divergence between distributions"""
        p = p + 1e-10
        q = q + 1e-10
        
        p = p / torch.sum(p)
        q = q / torch.sum(q)
        
        return torch.sum(p * torch.log(p / q))
        
    def forward(self, data, time, p=0):
        """Forward pass with improved tensor dimension handling"""
        batch_size, window_size, feature_num = data.shape
        
        # Apply disturbance if needed
        if p > 0:
            disturb = torch.rand(batch_size, feature_num, device=self.device) * p
            disturb = disturb.unsqueeze(1).repeat(1, window_size, 1)
            data_disturbed = data + disturb
        else:
            data_disturbed = data
            disturb = 0
        
        # Extract physical features with TFD calculation
        try:
            ph_high, ph_low, complexity, tfd = self.feature_extractor(
                data_disturbed, compute_tfd=True, batch_processing=True
            )
            
            # Initialize or update prior TFD
            if self.prior_tfd is None:
                self.prior_tfd = tfd.detach()
                
            # Compute KL divergence
            tfd_kl_div = self.compute_kl_divergence(tfd, self.prior_tfd)
                
        except Exception as e:
            print(f"Feature extraction failed: {str(e)}")
            ph_high = torch.zeros((batch_size, feature_num, window_size), device=self.device)
            ph_low = torch.zeros((batch_size, feature_num, window_size), device=self.device)
            complexity = torch.tensor(0.5, device=self.device)
            tfd_kl_div = torch.tensor(0.0, device=self.device)
        
        # Project physical features
        ph_high_emb = self.high_freq_proj(ph_high.transpose(1, 2).unsqueeze(-1))
        ph_low_emb = self.low_freq_proj(ph_low.transpose(1, 2).unsqueeze(-1))

        if ph_low.dim() > 3:
            ph_low = ph_low.view(batch_size, feature_num, window_size)
            
        if ph_low.shape[1] != feature_num or ph_low.shape[2] != window_size:
            if ph_low.shape[1] == window_size and ph_low.shape[2] == feature_num:
                ph_low = ph_low.transpose(1, 2)
            else:
                ph_low = ph_low.reshape(batch_size, feature_num, window_size)
        
        try:
            bt = torch.full((batch_size,), self.t, device=self.device)
            sample_noise = torch.randn_like(data_disturbed)
            noise_data = self.diffusion.q_sample(data_disturbed, ph_low, bt, sample_noise)
            
            if noise_data.dim() > 3:
                noise_data = noise_data.view(batch_size, window_size, feature_num)
            
            if self.use_langevin:
                def energy_fn(x):
                    return self.compute_energy_function(x, ph_low.transpose(1, 2))
                
                noise_data = self.diffusion.langevin_sample(
                    noise_data, energy_fn, n_steps=self.langevin_steps, step_size=self.langevin_step_size
                )
                
        except Exception as e:
            print(f"Diffusion process failed: {str(e)}")
            noise_data = data_disturbed
        
        # Apply model processing
        data_emb = self.data_embedding(noise_data) + self.position_embedding(noise_data)
        time_emb = self.time_embedding(time)
        x = data_emb + time_emb
        
        for block in self.encoder_blocks:
            x = block(x, ph_high_emb, ph_low_emb)
            
        recon = self.reconstruction_head(x)
        score = self.anomaly_predictor(x)
        
        # Remove disturbance if applied
        if isinstance(disturb, torch.Tensor):
            recon = recon - disturb
            
        return recon, score, complexity, tfd_kl_div
        
    def compute_loss(self, data, time, stable=None, label=None, p=0, lambda_aspe=0.1, lambda_energy=0.1):
        """Complete loss function with physical constraints"""
        # Forward pass
        recon, score, complexity, tfd_kl_div = self.forward(data, time, p)
        
        # Compute losses
        recon_loss = F.mse_loss(recon, data)
        aspe_loss = complexity
        energy_loss = tfd_kl_div
        
        # Detection loss if labels available
        if label is not None:
            detection_loss = F.binary_cross_entropy_with_logits(score, label)
        else:
            detection_loss = torch.tensor(0.0, device=self.device)
            
        # Combine losses according to theory
        total_loss = recon_loss + lambda_aspe * aspe_loss + lambda_energy * energy_loss
        if label is not None:
            total_loss += detection_loss
            
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'aspe_loss': aspe_loss,
            'energy_loss': energy_loss,
            'detection_loss': detection_loss,
            'tfd_kl_div': tfd_kl_div
        }
        
    def detect_anomalies(self, data, time, threshold=None):
        """Improved anomaly detection with physics-informed scoring"""
        with torch.no_grad():
            # Forward pass
            recon, score, complexity, tfd_kl_div = self.forward(data, time, p=0)
            
            # Compute metrics
            recon_error = F.mse_loss(recon, data, reduction='none').mean(dim=2)
            anomaly_score = score.squeeze(-1)
            
            # Combined score with theoretical components
            combined_score = (
                anomaly_score + 
                complexity * recon_error + 
                0.1 * tfd_kl_div.unsqueeze(0).unsqueeze(0).expand_as(anomaly_score)
            )
            
            # Threshold determination
            if threshold is None:
                mean = combined_score.mean()
                std = combined_score.std()
                threshold = mean + 3 * std
                
            predictions = (combined_score > threshold).float()
            
            return {
                'anomaly_score': combined_score,
                'predictions': predictions,
                'threshold': threshold,
                'reconstruction': recon,
                'reconstruction_error': recon_error,
                'complexity': complexity,
                'tfd_kl_div': tfd_kl_div
            }