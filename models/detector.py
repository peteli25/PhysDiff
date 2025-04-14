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
    """
    Extracts physical features from time series data using MAFD decomposition
    """
    def __init__(self, n_components=8, dic_dist=0.05, max_mag=0.95, device=None):
        super(PhysicalFeatureExtractor, self).__init__()
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Initialize MAFD
        self.mafd = MAFD(device=self.device)
        self.n_components = n_components
        self.dic_dist = dic_dist
        self.max_mag = max_mag
        
        # Initialize ASPE
        self.aspe = MultiScaleASPE(device=self.device)
        
    def _prepare_signal(self, x):
        """Prepare signal for MAFD"""
        # Transpose from (batch_size, seq_len, features) to (features, seq_len)
        # For single batch item
        if len(x.shape) == 3:
            x = x[0].transpose(0, 1)
        return x
        
    def _extract_components(self, mafd_results):
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
    
    def forward(self, x):
        """
        Extract physical features from time series data
        
        Parameters:
        -----------
        x : torch.Tensor
            Input time series data of shape (batch_size, seq_len, features)
            
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor, float]:
            high_freq: High frequency components
            low_freq: Low frequency components
            complexity: Signal complexity measure
        """
   
        batch_size = x.shape[0]
        
        all_high_freq = []
        all_low_freq = []
        all_complexity = []
        
        for b in range(batch_size):
            x_signal = self._prepare_signal(x[b:b+1])
            
            self.mafd.loadInputSignal(x_signal)
            self.mafd.genDic(dist=self.dic_dist, max_an_mag=self.max_mag)
            self.mafd.genEva()
            self.mafd.init_decomp()
            
            for _ in range(self.n_components - 1):
                self.mafd.nextDecomp()
            
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
            
            if orig_data.ndim > 1 and orig_data.shape[0] > 0:
                first_channel = orig_data[0]
            else:
                first_channel = orig_data
                
            complexity = self.aspe.compute_mean(first_channel)
            
            all_high_freq.append(high_freq)
            all_low_freq.append(low_freq)
            all_complexity.append(complexity)
        
        if batch_size > 1:
            try:
                stacked_high_freq = torch.stack(all_high_freq, dim=0)
                stacked_low_freq = torch.stack(all_low_freq, dim=0)
            except:
                stacked_high_freq = all_high_freq[0].unsqueeze(0).repeat(batch_size, 1, 1)
                stacked_low_freq = all_low_freq[0].unsqueeze(0).repeat(batch_size, 1, 1)
                
            avg_complexity = sum(all_complexity) / len(all_complexity)
            
            return stacked_high_freq, stacked_low_freq, avg_complexity
        else:
            return all_high_freq[0], all_low_freq[0], all_complexity[0]


class RoutingAttention(nn.Module):
    """
    Routing attention mechanism for physically-guided diffusion
    """
    def __init__(self, model_dim, atten_dim, head_num, dropout, residual=True):
        super(RoutingAttention, self).__init__()
        self.atten_dim = atten_dim
        self.head_num = head_num
        self.residual = residual
        self.model_dim = model_dim
        
        # Query, Key, Value projections for data
        self.W_Q = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_K = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_V = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        
        # Projections for physical features
        self.W_Ph_high = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_Ph_low = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        
        self.fc = nn.Linear(self.atten_dim * self.head_num, model_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)
        
    def forward(self, Q, K, V, Ph_high, Ph_low):
        """
        Forward pass for routing attention
        """
        residual = Q.clone()
        batch_size, seq_len = Q.size(0), Q.size(1)
        
        # Project Q, K, V
        Q = self.W_Q(Q).view(batch_size, seq_len, self.head_num, self.atten_dim)
        K = self.W_K(K).view(batch_size, seq_len, self.head_num, self.atten_dim)
        V = self.W_V(V).view(batch_size, seq_len, self.head_num, self.atten_dim)
        
        # Reshape physical features to match sequence length
        if Ph_high.size(1) != seq_len:
            # Reshape physical features to match batch_size and seq_len
            Ph_high = Ph_high.view(batch_size, -1, self.model_dim)
            # Either truncate or pad to match seq_len
            if Ph_high.size(1) > seq_len:
                Ph_high = Ph_high[:, :seq_len, :]
            elif Ph_high.size(1) < seq_len:
                padding = torch.zeros(batch_size, seq_len - Ph_high.size(1), self.model_dim, device=Ph_high.device)
                Ph_high = torch.cat([Ph_high, padding], dim=1)
        
        if Ph_low.size(1) != seq_len:
            # Same for Ph_low
            Ph_low = Ph_low.view(batch_size, -1, self.model_dim)
            if Ph_low.size(1) > seq_len:
                Ph_low = Ph_low[:, :seq_len, :]
            elif Ph_low.size(1) < seq_len:
                padding = torch.zeros(batch_size, seq_len - Ph_low.size(1), self.model_dim, device=Ph_low.device)
                Ph_low = torch.cat([Ph_low, padding], dim=1)
        
        # Project physical features
        Ph_high = self.W_Ph_high(Ph_high).view(batch_size, seq_len, self.head_num, self.atten_dim)
        Ph_low = self.W_Ph_low(Ph_low).view(batch_size, seq_len, self.head_num, self.atten_dim)
        
        # Transpose for attention calculation [batch_size, head_num, seq_len, atten_dim]
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        Ph_high = Ph_high.permute(0, 2, 1, 3)
        Ph_low = Ph_low.permute(0, 2, 1, 3)
        
        # Calculate attention scores
        scores_data = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        scores_high = torch.matmul(Q, Ph_high.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        scores_low = torch.matmul(Q, Ph_low.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        
        # Combine scores with proper weighting using sigmoid gates
        gate_high = torch.sigmoid(scores_high.mean(dim=-1, keepdim=True))
        gate_low = torch.sigmoid(scores_low.mean(dim=-1, keepdim=True))
        
        # Combined attention
        attn = F.softmax(scores_data + gate_high * scores_high + gate_low * scores_low, dim=-1)
        context = torch.matmul(attn, V)
        
        # Transpose back and reshape [batch_size, seq_len, head_num*atten_dim]
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        
        # Final projection
        output = self.dropout(self.fc(context))
        
        if self.residual:
            return self.norm(output + residual)
        else:
            return self.norm(output)


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


class PhysicallyGuidedDiffusion(nn.Module):
    """
    Physically-guided diffusion model for time series anomaly detection
    """
    def __init__(self, time_steps=1000, beta_start=0.0001, beta_end=0.02, 
                 window_size=64, model_dim=512, ff_dim=2048, atten_dim=64,
                 feature_num=8, time_num=5, block_num=2, head_num=8,
                 dropout=0.1, device=None, d=30, t=500, mafd_components=8):
        super(PhysicallyGuidedDiffusion, self).__init__()
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Model parameters
        self.window_size = window_size
        self.model_dim = model_dim
        self.feature_num = feature_num
        self.time_num = time_num
        self.t = t
        
        # Initialize feature extractor
        self.feature_extractor = PhysicalFeatureExtractor(
            n_components=mafd_components, 
            device=self.device
        )
        
        # Diffusion model
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
        
        # Physical feature projections
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
            
        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, feature_num)
        )
        
        # Anomaly score predictor
        self.anomaly_predictor = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 1)
        )
        
    def _prepare_physical_features(self, ph_high, ph_low, batch_size):
        """
        Prepare physical features for the model
        
        Parameters:
        -----------
        ph_high : torch.Tensor
            High frequency components with shape [channels, timesteps] or [batch_size, channels, timesteps]
        ph_low : torch.Tensor
            Low frequency components with shape [channels, timesteps] or [batch_size, channels, timesteps]
        batch_size : int
            Batch size for expansion
            
        Returns:
        --------
        Tuple: Properly shaped physical features
        """
        # Handle high frequency features
        if ph_high.dim() == 2:  # [channels, timesteps]
            # Interpolate to match window_size if needed
            if ph_high.size(1) != self.window_size:
                ph_high = torch.nn.functional.interpolate(
                    ph_high.unsqueeze(0), 
                    size=self.window_size, 
                    mode='linear'
                ).squeeze(0)
                
            # Repeat for each item in batch
            ph_high = ph_high.unsqueeze(0).repeat(batch_size, 1, 1)
        elif ph_high.size(0) != batch_size:
            # If first dimension is not batch_size, adjust
            ph_high = ph_high.repeat(batch_size, 1, 1)
            
        # Handle low frequency features
        if ph_low.dim() == 2:  # [channels, timesteps]
            if ph_low.size(1) != self.window_size:
                ph_low = torch.nn.functional.interpolate(
                    ph_low.unsqueeze(0), 
                    size=self.window_size, 
                    mode='linear'
                ).squeeze(0)
                
            ph_low = ph_low.unsqueeze(0).repeat(batch_size, 1, 1)
        elif ph_low.size(0) != batch_size:
            # If first dimension is not batch_size, adjust
            ph_low = ph_low.repeat(batch_size, 1, 1)
            
        # Ensure channel dimension matches feature_num for diffusion model
        # This is critical for the dimension mismatch error
        if ph_high.size(1) == 1 and self.feature_num > 1:
            ph_high = ph_high.repeat(1, self.feature_num, 1)
            
        if ph_low.size(1) == 1 and self.feature_num > 1:
            ph_low = ph_low.repeat(1, self.feature_num, 1)
            
        return ph_high, ph_low
        
    def forward(self, data, time, p=0):
        """
        Forward pass
        
        Parameters:
        -----------
        data : torch.Tensor
            Input data of shape (batch_size, window_size, feature_num)
        time : torch.Tensor
            Time features of shape (batch_size, window_size, time_num)
        p : float
            Disturbance magnitude for training (default: 0)
            
        Returns:
        --------
        Tuple:
            recon: Reconstructed data
            score: Anomaly scores
            complexity: Signal complexity
        """
        batch_size, window_size, feature_num = data.shape
        
        if p > 0:
            disturb = torch.rand(batch_size, feature_num, device=self.device) * p
            disturb = disturb.unsqueeze(1).repeat(1, window_size, 1)
            data_disturbed = data + disturb
        else:
            data_disturbed = data
            disturb = 0
        
        # batch_idx = torch.randint(0, batch_size, (1,)).item()
        # sample_data = data_disturbed[batch_idx:batch_idx+1]
        sample_data = data_disturbed[torch.randperm(batch_size)[:1]]
        
        try:
            ph_high, ph_low, complexity = self.feature_extractor(sample_data)

            if len(ph_high.shape) == 2: 
                ph_high = ph_high.unsqueeze(0) 
            if len(ph_low.shape) == 2: 
                ph_low = ph_low.unsqueeze(0)
                
            if ph_high.shape[0] == 1 and batch_size > 1:
                ph_high = ph_high.expand(batch_size, -1, -1)
            if ph_low.shape[0] == 1 and batch_size > 1:
                ph_low = ph_low.expand(batch_size, -1, -1)
        except Exception as e:
            print(f"Feature extraction failed: {str(e)}")
            ph_high = torch.zeros((batch_size, 1, window_size), device=self.device)
            ph_low = torch.zeros((batch_size, 1, window_size), device=self.device)
            complexity = 0.5
        
        if ph_high.dim() > 3:
            ph_high = ph_high.view(batch_size, -1, window_size)
        if ph_low.dim() > 3:
            ph_low = ph_low.view(batch_size, -1, window_size)
        
        ph_high_emb = self.high_freq_proj(ph_high.unsqueeze(-1))
        ph_low_emb = self.low_freq_proj(ph_low.unsqueeze(-1))
        
        bt = torch.full((batch_size,), self.t, device=self.device)
        sample_noise = torch.randn_like(data_disturbed)
        
        try:
            noise_data = self.diffusion.q_sample(data_disturbed, ph_low, bt, sample_noise)
        except Exception as e:
            print(f"Diffusion sampling failed: {str(e)}")
            noise_data = data_disturbed
        
        data_emb = self.data_embedding(noise_data) + self.position_embedding(noise_data)
        time_emb = self.time_embedding(time)
        x = data_emb + time_emb
        
        for block in self.encoder_blocks:
            x = block(x, ph_high_emb, ph_low_emb)
            
        recon = self.reconstruction_head(x)
        score = self.anomaly_predictor(x)
        
        if isinstance(disturb, torch.Tensor):
            recon = recon - disturb
            
        return recon, score, complexity
        
    def compute_loss(self, data, time, stable=None, label=None, p=0, lambda_aspe=0.1):
        """
        Compute training loss
        
        Parameters:
        -----------
        data : torch.Tensor
            Input data
        time : torch.Tensor
            Time features
        stable : torch.Tensor
            Stable component (if available)
        label : torch.Tensor
            Ground truth labels (if available)
        p : float
            Disturbance magnitude
        lambda_aspe : float
            Weight for ASPE loss term
            
        Returns:
        --------
        Dict: Loss components
        """
        # Forward pass
        recon, score, complexity = self.forward(data, time, p)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, data)
        
        # ASPE regularization loss
        aspe_loss = complexity
        
        # Anomaly detection loss if labels available
        if label is not None:
            # Convert scores to binary predictions
            pred = (score > 0.5).float()
            # Compute BCE loss
            detection_loss = F.binary_cross_entropy_with_logits(score, label)
        else:
            detection_loss = torch.tensor(0.0, device=self.device)
            
        # Combine losses
        total_loss = recon_loss + lambda_aspe * aspe_loss
        if label is not None:
            total_loss += detection_loss
            
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'aspe_loss': aspe_loss,
            'detection_loss': detection_loss
        }
        
    def detect_anomalies(self, data, time, threshold=None):
        """
        Detect anomalies in the data
        
        Parameters:
        -----------
        data : torch.Tensor
            Input data
        time : torch.Tensor
            Time features
        threshold : float, optional
            Anomaly threshold. If None, uses statistical approach
            
        Returns:
        --------
        Dict: Anomaly detection results
        """
        with torch.no_grad():
            recon, score, complexity = self.forward(data, time, p=0)
            
            recon_error = F.mse_loss(recon, data, reduction='none')
            recon_error = recon_error.mean(dim=2)
            
            anomaly_score = score.squeeze(-1)
            
            combined_score = anomaly_score + complexity * recon_error
            
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
                'complexity': complexity
            }