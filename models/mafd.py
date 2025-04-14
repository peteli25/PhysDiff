import torch
import torch.fft
import numpy as np
import time
from scipy import signal
import warnings
from math import pi
from typing import Union, Optional, Dict, List, Tuple
import os

class MAFD:
    """
    Multi-channel Adaptive Fourier Decomposition implemented in PyTorch with GPU support
    """
    def __init__(self, device: Optional[str] = None):
        """
        Initialize MAFD calculation

        Parameters
        -----------------
        device : Optional[str]
            Device to use for computation ('cpu' or 'cuda')
        """
        # Automatically select device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        self.initSettings()
    
    def initSettings(self):
        """
        Set all parameters to default values
        """
        self.s = []  # Original input signal
        self.G = []  # Analytic representation of the original input signal "s"
        self.t = []  # Phase of the original input signal "s"
        self.S1 = []  # Energy distribution
        self.max_loc = []  # Location of maximum energy
        self.weight = []  # Weight for computing numerical integration
        self.an = []  # Searching results of the basis parameter
        self.coef = []  # Decomposition coefficients 
        self.level = 0  # Decomposition level (initial level is 0)
        self.dic_an = []  # Searching dictionary of an
        self.dic_an_search = []  # Searching dictionary for Fast AFD
        self.Base = []  # Evaluators of searching an
        self.remainder = []  # Decomposition remainder
        self.tem_B = []  # Decomposition basis components
        self.deComp = []  # Decomposition components
        self.decompMethod = 4  # Multi-channel Fast AFD
        self.dicGenMethod = 2  # Dictionary generation methods: Circle (Fast AFD must be "circle")
        self.AFDMethod = 1  # AFD methods: core (default)
        self.run_time = []  # Running time of decomposition
        self.time_genDic = 0  # Running time of generating searching dictionary
        self.time_genEva = 0  # Running time of generating evaluators
        
    def loadInputSignal(self, input_signal: Union[np.ndarray, str, torch.Tensor]):
        """
        Load input signal

        Parameters
        ----------------------
        input_signal : Union[np.ndarray, str, torch.Tensor] 
            The input signal. Multi-channel signal.
            The format can be:
                + numpy array/torch.Tensor: Input signal. The dimension must be C * N where C is channels, N is the total sampling number.
                + String: File path storing the input signal. Current supporting file format:
                    - ".mat": matlab file. Signal is stored in a matrix called "G".
                    - ".npy": numpy file. Signal is stored in a numpy array called "G".
        """
        # Load signal
        if isinstance(input_signal, str):
            # Check whether file exists
            if not os.path.exists(input_signal):
                raise ValueError("The provided input file location does not exist!!")
            
            # Load signal from file
            file_name, file_extension = os.path.splitext(input_signal)
            if len(file_extension) == 0:
                raise ValueError("Cannot get the extension of the input file!!")
                
            if file_extension[1:].lower() == 'mat':
                data = self._loadmat(input_signal)
                s = data['G']
            elif file_extension[1:].lower() == 'npy':
                data = np.load(input_signal, allow_pickle=True).item()
                s = data['G']
            else:
                raise ValueError("Unknown extension of the input file!!")
        elif isinstance(input_signal, np.ndarray):
            s = input_signal.copy()
        elif isinstance(input_signal, torch.Tensor):
            s = input_signal.cpu().numpy() if input_signal.device != torch.device('cpu') else input_signal.numpy()
        else:
            raise ValueError("The type of 'input_signal' must be a numpy array, torch.Tensor, or a string")
        
        # Check dimension
        if len(s.shape) == 1:
            s = np.expand_dims(s, axis=0)
        if len(s.shape) != 2:
            raise ValueError("The dimension of the input signal must be C*N !!")
            
        N_ch, N_sample = s.shape
        if N_sample == 1:
            s = s.T
            
        # Store signal
        self.s = torch.tensor(s, dtype=torch.float32, device=self.device)
        
        # Hilbert transform
        self.G = torch.zeros(*self.s.shape, dtype=torch.complex64, device=self.device)
        
        # Perform Hilbert transform channel by channel
        for i_ch in range(self.s.shape[0]):
            # Convert to numpy for Hilbert transform
            s_np = self.s[i_ch].cpu().numpy()
            analytic_signal = signal.hilbert(s_np)
            # Convert back to PyTorch tensor
            self.G[i_ch, :] = torch.tensor(analytic_signal, dtype=torch.complex64, device=self.device)
        
        # Set phase
        N_ch, N_sample = self.G.shape
        t = torch.arange(0, N_sample, device=self.device) / N_sample * 2 * pi
        t = t.unsqueeze(0)
        self.t = t.repeat(N_ch, 1)
        
        # Generate weights (default: uniform weights)
        self.weight = self._genWeight(N_sample)
    
    def _loadmat(self, file_path: str) -> dict:
        """
        Load mat file

        Parameters
        -------------------
        file_path: str
            Full file path

        Returns
        -------
        mat_data: dict
            Data in mat file
        """
        try:
            import scipy.io as sio
            data = sio.loadmat(file_path)
            # Simple processing for basic MATLAB files
            return data
        except:
            raise ValueError("Cannot load the MAT file. Try using scipy.io.loadmat directly.")
    
    def _genWeight(self, N: int) -> torch.Tensor:
        """
        Generate weights for numerical integration

        Parameters
        ----------
        N : int
            Number of sample points

        Returns
        -------
        weight : torch.Tensor
            Weights for numerical integration
        """
        return torch.ones((N, 1), dtype=torch.complex64, device=self.device)
    
    def _Circle_Disk(self, dist: float, max_an_mag: float, sig_len: int, max_an_phase: float) -> torch.Tensor:
        """
        Generate circle searching dictionary

        Parameters
        -------------
        dist : float
            Distance between two adjacent magnitude values
        max_an_mag : float
            Maximum magnitude of an
        sig_len : int
            Signal length
        max_an_phase : float
            Maximum phase of an
        """
        phase_a = torch.arange(0.0, max_an_phase + 2*pi/sig_len, 2*pi/sig_len, device=self.device)
        phase_a = phase_a.unsqueeze(0)
        abs_a = torch.arange(0.0, 1.0 + dist, dist, device=self.device)
        abs_a = abs_a[:-1]  # Remove the last element (1.0 exactly)
        abs_a = abs_a.unsqueeze(0)
        
        _, n_phase = phase_a.shape
        _, n_abs = abs_a.shape
        abs_a = abs_a.repeat(n_phase, 1).T
        phase_a = phase_a.repeat(n_abs, 1)

        # Create complex values using magnitude and phase
        ret1 = abs_a * torch.exp(1j * phase_a)
        
        # Create mask for valid values (magnitude less than max_an_mag)
        mask = (torch.abs(ret1) - max_an_mag >= -1e-15)
        
        # Apply mask to ret1 (NaN for invalid values)
        ret1_numpy = ret1.cpu().numpy()
        ret1_numpy[mask.cpu().numpy()] = np.nan
        
        # Remove rows that are all NaN
        row_nan_mask = np.isnan(ret1_numpy).all(axis=1)
        ret2_numpy = ret1_numpy[~row_nan_mask]
        
        # Remove columns that are all NaN
        col_nan_mask = np.isnan(ret2_numpy).all(axis=0)
        ret3_numpy = ret2_numpy[:, ~col_nan_mask]
        
        # Convert back to PyTorch tensor
        ret3 = torch.tensor(ret3_numpy, dtype=torch.complex64, device=self.device)
        
        return ret3
    
    def genDic(self, dist: float, max_an_mag: float):
        """
        Generate searching dictionary

        Parameters
        -------------
        dist : float
            Distance between two adjacent magnitude values
        max_an_mag: float
            Maximum magnitude of an
        """
        # Check inputs
        if dist < 0 or dist > 1:
            raise ValueError("The distance between two adjacent magnitude values must be within 0~1")
        if max_an_mag < 0 or max_an_mag > 1:
            raise ValueError("The maximum magnitude must be within 0~1")
        if len(self.s) == 0:
            raise ValueError("Please load input signal first")
            
        # Generate searching dictionary
        start_time = time.time()
        N_ch, N_sample = self.G.shape
        
        # Fast AFD with Circle dictionary
        if self.decompMethod == 4:  # Multi-channel Fast AFD
            if self.dicGenMethod == 1:
                raise ValueError("The fast AFD cannot use the 'square' dictionary.")
            elif self.dicGenMethod == 2:
                self.dic_an = [self._Circle_Disk(dist, max_an_mag, N_sample, 0) for _ in range(N_ch)]
                self.dic_an_search = [self._Circle_Disk(dist, max_an_mag, N_sample, 2*pi-2*pi/N_sample) for _ in range(N_ch)]
            else:
                raise ValueError('Unknown dicGenMethod')
        else:
            raise ValueError('This implementation only supports decompMethod=4 (Multi-channel Fast AFD)')
            
        self.time_genDic = time.time() - start_time
    
    def _e_a(self, a, t):
        """
        Evaluator function

        Parameters
        ----------
        a : complex
            Parameter a
        t : torch.Tensor
            Time/phase vector

        Returns
        -------
        torch.Tensor
            Evaluator values
        """
        return ((1 - torch.abs(a)**2) ** 0.5) / (1 - torch.conj(a) * torch.exp(1j * t))
    
    def genEva(self):
        """
        Generate evaluators
        """
        # Check dictionary
        if len(self.dic_an) == 0:
            raise ValueError("Please generate the searching dictionary first!!")

        start_time = time.time()

        # Initialize evaluator
        N_ch, N_sample = self.G.shape
        self.Base = []
        
        # Generate evaluators for Multi-channel Fast AFD
        if self.decompMethod == 4:
            for i_ch in range(N_ch):
                dic_row, dic_col = self.dic_an[i_ch].shape
                base_ch = torch.zeros((dic_row, dic_col, N_sample), dtype=torch.complex64, device=self.device)
                
                for i in range(dic_row):
                    for j in range(dic_col):
                        if not torch.isnan(self.dic_an[i_ch][i, j]):
                            # For Fast AFD, we use FFT of the evaluator
                            ea = self._e_a(self.dic_an[i_ch][i, j], self.t[[i_ch], :])
                            base_ch[i, j, :] = torch.fft.fft(ea, N_sample)
                            
                self.Base.append(base_ch)
        else:
            raise ValueError('This implementation only supports decompMethod=4 (Multi-channel Fast AFD)')

        self.time_genEva = time.time() - start_time
    
    def init_decomp(self):
        """
        Initialize decomposition
        """
        N_ch, _ = self.G.shape
        # Remove historical decomposition
        self.S1 = [[] for _ in range(N_ch)]  # Energy distribution
        self.max_loc = [[] for _ in range(N_ch)]  # Location of maximum energy
        self.an = [[] for _ in range(N_ch)]  # Searching results of the basis parameter
        self.coef = [[] for _ in range(N_ch)]  # Decomposition coefficients 
        self.level = 0  # Decomposition level (initial level is 0)
        self.remainder = [[] for _ in range(N_ch)]  # Decomposition remainder
        for i_ch in range(N_ch):
            self.remainder[i_ch].append(self.G[[i_ch], :].clone())
        self.tem_B = [[] for _ in range(N_ch)]  # Decomposition basis components
        self.deComp = [[] for _ in range(N_ch)]  # Decomposition components
        self.run_time = []  # Running time of decomposition

        start_time = time.time()

        for i_ch in range(N_ch):
            # Initial stage: a_0=0
            self.S1[i_ch].append(None)
            self.max_loc[i_ch].append(None)
            an = torch.tensor(0.0, dtype=torch.complex64, device=self.device)
            self.an[i_ch].append(an)
            
            # Decomposition coefficient
            coef = self._calCoef(an, self.t[[i_ch], :], self.remainder[i_ch][self.level], self.weight)
            self.coef[i_ch].append(coef)
            
            # Basis component
            tem_B = (torch.sqrt(1-torch.abs(an)**2) / (1-torch.conj(an) * torch.exp(self.t[[i_ch], :] * 1j)))
            self.tem_B[i_ch].append(tem_B)
            
            # Decomposition component
            deComp = self.coef[i_ch][self.level] * self.tem_B[i_ch][self.level]
            self.deComp[i_ch].append(deComp)
            
            # Remainder
            remainder = (self.remainder[i_ch][self.level] - coef * self._e_a(an, self.t[[i_ch], :])) * \
                       (1 - torch.conj(an) * torch.exp(1j * self.t[[i_ch], :])) / \
                       (torch.exp(1j * self.t[[i_ch], :]) - an)
            self.remainder[i_ch].append(remainder)

        self.run_time.append(time.time() - start_time)
    
    def _calCoef(self, a, t, G, W):
        """
        Calculate coefficient for decomposition

        Parameters
        ----------
        a : complex
            Parameter a
        t : torch.Tensor
            Time/phase vector
        G : torch.Tensor
            Signal
        W : torch.Tensor
            Weights

        Returns
        -------
        torch.Tensor
            Coefficient
        """
        return torch.conj(self._e_a(a, t).matmul((G.conj().T * W))) / G.shape[1]
    
    def nextDecomp(self):
        """
        Next decomposition level
        """
        N_ch, _ = self.G.shape
        self.level += 1
        start_time = time.time()
        
        # For Multi-channel Fast AFD
        if self.decompMethod == 4:
            # Calculate S1 (energy distribution) for all channels
            S1_tmp_sum = None
            
            for i_ch in range(N_ch):
                Base = self.Base[i_ch][0, :, :]
                
                # Use FFT for efficient calculation of inner product
                fft_remainder = torch.fft.fft(self.remainder[i_ch][self.level] * self.weight.T, self.t.shape[1])
                fft_remainder_repeated = fft_remainder.repeat(Base.shape[0], 1)
                
                # Element-wise multiplication in frequency domain
                S1_tmp = torch.abs(torch.fft.ifft(fft_remainder_repeated * Base, self.t.shape[1], dim=1))
                S1_tmp = S1_tmp.T
                
                if S1_tmp_sum is None:
                    S1_tmp_sum = S1_tmp.clone()
                else:
                    S1_tmp_sum = S1_tmp_sum + S1_tmp
                    
            # Average over all channels
            S1_tmp = S1_tmp_sum / N_ch
            
            # Find the maximum energy point
            for i_ch in range(N_ch):
                if i_ch == 0:
                    self.S1[i_ch].append(S1_tmp)
                    # Find the max location
                    max_val = torch.max(S1_tmp)
                    max_indices = torch.where(S1_tmp == max_val)
                    max_row, max_col = max_indices[0][0].item(), max_indices[1][0].item()
                    self.max_loc[i_ch].append((max_row, max_col))
                    
                    # Get the corresponding an
                    dic_an = self.dic_an_search[i_ch]
                    an = dic_an[max_row, max_col]
                    self.an[i_ch].append(an)
                    
                    # Decomposition coefficient
                    coef = self._calCoef(an, self.t[[i_ch], :], self.remainder[i_ch][self.level], self.weight)
                    self.coef[i_ch].append(coef)
                    
                    # Basis component
                    tem_B = (torch.sqrt(1-torch.abs(an)**2) / (1-torch.conj(an) * torch.exp(self.t[[i_ch], :] * 1j))) * \
                           ((torch.exp(1j * self.t[[i_ch], :]) - self.an[i_ch][self.level-1]) / \
                           (torch.sqrt(1 - torch.abs(self.an[i_ch][self.level-1])**2))) * \
                           self.tem_B[i_ch][self.level-1]
                    self.tem_B[i_ch].append(tem_B)
                    
                    # Decomposition component
                    deComp = self.coef[i_ch][self.level] * self.tem_B[i_ch][self.level]
                    self.deComp[i_ch].append(deComp)
                    
                    # Remainder
                    remainder = (self.remainder[i_ch][self.level] - coef * self._e_a(an, self.t[[i_ch], :])) * \
                               (1 - torch.conj(an) * torch.exp(1j * self.t[[i_ch], :])) / \
                               (torch.exp(1j * self.t[[i_ch], :]) - an)
                    self.remainder[i_ch].append(remainder)
                else:
                    # For other channels, use the same an and max_loc as channel 0
                    self.S1[i_ch].append(self.S1[0][-1])
                    self.max_loc[i_ch].append(self.max_loc[0][-1])
                    self.an[i_ch].append(self.an[0][-1])
                    
                    # Decomposition coefficient
                    coef = self._calCoef(self.an[0][-1], self.t[[i_ch], :], self.remainder[i_ch][self.level], self.weight)
                    self.coef[i_ch].append(coef)
                    
                    # Basis component
                    tem_B = (torch.sqrt(1-torch.abs(self.an[0][-1])**2) / (1-torch.conj(self.an[0][-1]) * torch.exp(self.t[[i_ch], :] * 1j))) * \
                           ((torch.exp(1j * self.t[[i_ch], :]) - self.an[i_ch][self.level-1]) / \
                           (torch.sqrt(1 - torch.abs(self.an[i_ch][self.level-1])**2))) * \
                           self.tem_B[i_ch][self.level-1]
                    self.tem_B[i_ch].append(tem_B)
                    
                    # Decomposition component
                    deComp = self.coef[i_ch][self.level] * self.tem_B[i_ch][self.level]
                    self.deComp[i_ch].append(deComp)
                    
                    # Remainder
                    remainder = (self.remainder[i_ch][self.level] - coef * self._e_a(self.an[0][-1], self.t[[i_ch], :])) * \
                               (1 - torch.conj(self.an[0][-1]) * torch.exp(1j * self.t[[i_ch], :])) / \
                               (torch.exp(1j * self.t[[i_ch], :]) - self.an[0][-1])
                    self.remainder[i_ch].append(remainder)
        else:
            raise ValueError('This implementation only supports decompMethod=4 (Multi-channel Fast AFD)')
            
        self.run_time.append(time.time() - start_time)
    
    def reconstruct(self, level):
        """
        Calculate reconstructed signal

        Parameters
        ----------
        level : int
            Decomposition level to reconstruct up to

        Returns
        -------
        torch.Tensor
            Reconstructed signal
        """
        if level > self.level:
            raise ValueError(f"The current decomposition level is {self.level}. "
                           f"If you want to reconstruct the signal using decomposition components "
                           f"in higher levels, please use 'nextDecomp()' to get more components.")
                           
        N_ch, _ = self.G.shape
        re_sig = torch.zeros_like(self.G)
        
        for i_ch in range(N_ch):
            select_deComp = self.deComp[i_ch][:level+1]
            re_sig[[i_ch], :] = select_deComp[0]
            
            k = 1
            while k < len(select_deComp):
                re_sig[[i_ch], :] = re_sig[[i_ch], :] + select_deComp[k]
                k += 1
                
        return re_sig.real
    
    def decomp(self, level):
        """
        Decompose from the initial decomposition to the given level.
        """
        if self.level >= level:
            warnings.warn("The current decomposition already includes the given level.")
        if self.level <= 0:
            self.init_decomp()
        while self.level < level:
            self.nextDecomp()


    # Add this new method to the MAFD class in mafd.py

    def decompose_batch(self, signals):
        """
        Batch processing for multiple signals
        
        Parameters:
        -----------
        signals : torch.Tensor
            Batch of input signals [batch_size, channels, timesteps]
            
        Returns:
        --------
        list: List of decomposition results for each signal
        """
        batch_size = signals.shape[0]
        results = []
        
        # Process in smaller batches if needed to avoid memory issues
        max_batch = 16
        for b in range(0, batch_size, max_batch):
            end = min(b + max_batch, batch_size)
            batch_signals = signals[b:end]
            batch_results = []
            
            for i in range(end - b):
                signal = batch_signals[i]
                
                # Load and decompose signal
                self.loadInputSignal(signal)
                self.genDic(dist=0.05, max_an_mag=0.95)
                self.genEva()
                self.init_decomp()
                
                for _ in range(self.n_components - 1 if hasattr(self, 'n_components') else 7):
                    self.nextDecomp()
                    
                # Get decomposition results
                orig_data = self.s.cpu().numpy()
                recon = self.reconstruct(self.level).cpu().numpy()
                
                # Extract components
                components = []
                for ch in range(len(self.deComp)):
                    components.append([comp.cpu().detach() if hasattr(comp, 'cpu') else comp 
                                    for comp in self.deComp[ch]])
                    
                # Package results
                result = {
                    'original': orig_data,
                    'reconstructed': recon,
                    'components': components,
                    'parameters': {
                        'a': [self.an[ch] for ch in range(len(self.an))],
                        'coef': [self.coef[ch] for ch in range(len(self.coef))],
                        'level': self.level
                    }
                }
                
                batch_results.append(result)
            
            results.extend(batch_results)
            
        return results