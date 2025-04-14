import numpy as np
import torch
from typing import Optional, Tuple, List, Union


class ASPE:
    """
    Amplitude-Sensitive Permutation Entropy for time series complexity measurement
    """
    def __init__(self, m: int = 3, tau: int = 1, device: Optional[str] = None):
        """
        Initialize ASPE calculation
        
        Parameters:
        -----------
        m : int
            Embedding dimension (default: 3)
        tau : int
            Time delay (default: 1)
        device : Optional[str]
            Device to use for computation ('cpu' or 'cuda')
        """
        self.m = m
        self.tau = tau
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Pre-compute factorial for efficiency
        self.factorial_m = np.math.factorial(self.m)
        
        # Generate all possible permutation patterns
        self.all_patterns = self._generate_patterns()
        
    def _generate_patterns(self) -> List[Tuple]:
        """
        Generate all possible permutation patterns of order m
        
        Returns:
        --------
        List[Tuple]: List of all possible permutation patterns
        """
        import itertools
        return list(itertools.permutations(range(self.m)))
    
    def _create_embedding_vectors(self, signal: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Create embedding vectors from signal
        
        Parameters:
        -----------
        signal : Union[np.ndarray, torch.Tensor]
            Input signal
        
        Returns:
        --------
        torch.Tensor: Embedding vectors of shape (n, m) where n is the number of vectors
        """
        # Convert to tensor if needed
        if isinstance(signal, np.ndarray):
            signal = torch.tensor(signal, dtype=torch.float32, device=self.device)
        elif isinstance(signal, torch.Tensor) and signal.device != self.device:
            signal = signal.to(self.device)
            
        # Flatten if multidimensional
        if len(signal.shape) > 1:
            signal = signal.reshape(-1)
            
        # Number of embedding vectors
        n = len(signal) - (self.m - 1) * self.tau
        
        if n <= 0:
            raise ValueError(f"Signal length {len(signal)} too short for embedding parameters m={self.m}, tau={self.tau}")
            
        # Create embedding vectors
        embedding_vectors = torch.zeros((n, self.m), dtype=torch.float32, device=self.device)
        
        for i in range(n):
            for j in range(self.m):
                embedding_vectors[i, j] = signal[i + j * self.tau]
                
        return embedding_vectors
    
    def _get_permutation_patterns(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Get permutation pattern indices for each embedding vector
        
        Parameters:
        -----------
        vectors : torch.Tensor
            Embedding vectors of shape (n, m)
        
        Returns:
        --------
        torch.Tensor: Pattern indices for each vector
        """
        n = vectors.shape[0]
        patterns = torch.zeros(n, dtype=torch.long, device=self.device)
        
        # Process vectors in batches for efficiency
        batch_size = 1000
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            batch = vectors[i:end]
            
            # Get ranks for each vector (argsort of argsort gives ranks)
            ranks = torch.zeros_like(batch)
            # Vectorized operation for getting ranks
            ranks = torch.argsort(torch.argsort(batch, dim=1), dim=1)
                
            # Convert ranks to pattern indices
            for j in range(end - i):
                pattern_tuple = tuple(ranks[j].cpu().numpy().astype(int))
                pattern_idx = self.all_patterns.index(pattern_tuple)
                patterns[i + j] = pattern_idx
                
        return patterns
    
    def _compute_amplitude_weights(self, vectors: torch.Tensor) -> torch.Tensor:
        """Compute amplitude weights using coefficient of variation"""
        # Coefficient of Variation = std / mean
        epsilon = 1e-8
        means = torch.mean(vectors, dim=1)
        stds = torch.std(vectors, dim=1)
        weights = stds / (torch.abs(means) + epsilon)
        return weights
    
    def compute(self, signal: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Compute Amplitude-Sensitive Permutation Entropy
        
        Parameters:
        -----------
        signal : Union[np.ndarray, torch.Tensor]
            Input signal
        
        Returns:
        --------
        float: ASPE value
        """
        # Create embedding vectors
        vectors = self._create_embedding_vectors(signal)
        
        # Get permutation patterns
        patterns = self._get_permutation_patterns(vectors)
        
        # Compute amplitude weights
        weights = self._compute_amplitude_weights(vectors)
        
        # Calculate weighted pattern probabilities
        pattern_probs = torch.zeros(self.factorial_m, dtype=torch.float32, device=self.device)
        
        # Vectorized computation of pattern probabilities
        for i in range(self.factorial_m):
            pattern_indices = (patterns == i)
            if torch.any(pattern_indices):
                pattern_probs[i] = torch.sum(weights[pattern_indices])
        
        # Normalize probabilities
        total_weight = torch.sum(pattern_probs)
        if total_weight > 0:
            pattern_probs = pattern_probs / total_weight
        
        # Calculate entropy
        # Only consider non-zero probabilities to avoid log(0)
        entropy = torch.tensor(0.0, device=self.device)
        non_zero_probs = pattern_probs[pattern_probs > 0]
        entropy = -torch.sum(non_zero_probs * torch.log(non_zero_probs))
        
        # Normalize by maximum entropy
        max_entropy = np.log(self.factorial_m)
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy.item()
    
    @staticmethod
    def compute_batch(signals: Union[np.ndarray, torch.Tensor], m: int = 3, tau: int = 1, 
                      device: Optional[str] = None) -> np.ndarray:
        """
        Compute ASPE for batch of signals
        
        Parameters:
        -----------
        signals : Union[np.ndarray, torch.Tensor]
            Batch of input signals of shape (batch_size, signal_length)
        m : int
            Embedding dimension
        tau : int
            Time delay
        device : Optional[str]
            Device to use for computation
            
        Returns:
        --------
        np.ndarray: ASPE values for each signal
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        aspe_calculator = ASPE(m=m, tau=tau, device=device)
        
        if isinstance(signals, torch.Tensor):
            batch_size = signals.shape[0]
        else:
            batch_size = signals.shape[0]
            
        results = np.zeros(batch_size)
        
        # Process in smaller batches if needed to avoid memory issues
        max_batch = 32
        for i in range(0, batch_size, max_batch):
            end = min(i + max_batch, batch_size)
            batch = signals[i:end]
            
            # Process each signal in the batch
            for j in range(end - i):
                results[i + j] = aspe_calculator.compute(batch[j])
            
        return results


class MultiScaleASPE:
    """
    Multi-scale Amplitude-Sensitive Permutation Entropy
    Computes ASPE at multiple scales (embedding dimensions and time delays)
    """
    def __init__(self, m_range: List[int] = [2, 3, 4], tau_range: List[int] = [1, 2, 3], 
                 device: Optional[str] = None):
        """
        Initialize Multi-scale ASPE calculation
        
        Parameters:
        -----------
        m_range : List[int]
            Range of embedding dimensions to use
        tau_range : List[int]
            Range of time delays to use
        device : Optional[str]
            Device to use for computation
        """
        self.m_range = m_range
        self.tau_range = tau_range
        self.device = device
        
        # Create ASPE calculators for each scale
        self.calculators = []
        for m in m_range:
            for tau in tau_range:
                self.calculators.append(ASPE(m=m, tau=tau, device=device))
    
    def compute(self, signal: Union[np.ndarray, torch.Tensor]) -> List[float]:
        """
        Compute ASPE at multiple scales
        
        Parameters:
        -----------
        signal : Union[np.ndarray, torch.Tensor]
            Input signal
            
        Returns:
        --------
        List[float]: ASPE values at each scale
        """
        results = []
        for calculator in self.calculators:
            results.append(calculator.compute(signal))
        return results
    
    def compute_mean(self, signal: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Compute mean ASPE across all scales
        
        Parameters:
        -----------
        signal : Union[np.ndarray, torch.Tensor]
            Input signal
            
        Returns:
        --------
        float: Mean ASPE value
        """
        values = self.compute(signal)
        return np.mean(values)