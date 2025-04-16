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
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        self.initSettings()
    
    def initSettings(self):
        self.s = []
        self.G = []
        self.t = []
        self.S1 = []
        self.max_loc = []
        self.weight = []
        self.an = []
        self.coef = []
        self.level = 0
        self.dic_an = []
        self.dic_an_search = []
        self.Base = []
        self.remainder = []
        self.tem_B = []
        self.deComp = []
        self.decompMethod = 4
        self.dicGenMethod = 2
        self.AFDMethod = 1
        self.run_time = []
        self.time_genDic = 0
        self.time_genEva = 0
        
    def loadInputSignal(self, input_signal: Union[np.ndarray, str, torch.Tensor]):
        if isinstance(input_signal, str):
            if not os.path.exists(input_signal):
                raise ValueError("The provided input file location does not exist!!")
            
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
        
        if len(s.shape) == 1:
            s = np.expand_dims(s, axis=0)
        if len(s.shape) != 2:
            raise ValueError("The dimension of the input signal must be C*N !!")
            
        N_ch, N_sample = s.shape
        if N_sample == 1:
            s = s.T
            
        self.s = torch.tensor(s, dtype=torch.float32, device=self.device)
        
        self.G = torch.zeros(*self.s.shape, dtype=torch.complex64, device=self.device)
        
        for i_ch in range(self.s.shape[0]):
            s_np = self.s[i_ch].cpu().numpy()
            analytic_signal = signal.hilbert(s_np)
            self.G[i_ch, :] = torch.tensor(analytic_signal, dtype=torch.complex64, device=self.device)
        
        N_ch, N_sample = self.G.shape
        t = torch.arange(0, N_sample, device=self.device) / N_sample * 2 * pi
        t = t.unsqueeze(0)
        self.t = t.repeat(N_ch, 1)
        
        self.weight = self._genWeight(N_sample)
    
    def _loadmat(self, file_path: str) -> dict:
        try:
            import scipy.io as sio
            data = sio.loadmat(file_path)
            return data
        except:
            raise ValueError("Cannot load the MAT file. Try using scipy.io.loadmat directly.")
    
    def _genWeight(self, N: int) -> torch.Tensor:
        return torch.ones((N, 1), dtype=torch.complex64, device=self.device)
    
    def _Circle_Disk(self, dist: float, max_an_mag: float, sig_len: int, max_an_phase: float) -> torch.Tensor:
        phase_a = torch.arange(0.0, max_an_phase + 2*pi/sig_len, 2*pi/sig_len, device=self.device)
        phase_a = phase_a.unsqueeze(0)
        abs_a = torch.arange(0.0, 1.0 + dist, dist, device=self.device)
        abs_a = abs_a[:-1]
        abs_a = abs_a.unsqueeze(0)
        
        _, n_phase = phase_a.shape
        _, n_abs = abs_a.shape
        abs_a = abs_a.repeat(n_phase, 1).T
        phase_a = phase_a.repeat(n_abs, 1)

        ret1 = abs_a * torch.exp(1j * phase_a)
        
        mask = (torch.abs(ret1) - max_an_mag >= -1e-15)
        
        ret1_numpy = ret1.cpu().numpy()
        ret1_numpy[mask.cpu().numpy()] = np.nan
        
        row_nan_mask = np.isnan(ret1_numpy).all(axis=1)
        ret2_numpy = ret1_numpy[~row_nan_mask]
        
        col_nan_mask = np.isnan(ret2_numpy).all(axis=0)
        ret3_numpy = ret2_numpy[:, ~col_nan_mask]
        
        ret3 = torch.tensor(ret3_numpy, dtype=torch.complex64, device=self.device)
        
        return ret3
    
    def genDic(self, dist: float, max_an_mag: float):
        if dist < 0 or dist > 1:
            raise ValueError("The distance between two adjacent magnitude values must be within 0~1")
        if max_an_mag < 0 or max_an_mag > 1:
            raise ValueError("The maximum magnitude must be within 0~1")
        if len(self.s) == 0:
            raise ValueError("Please load input signal first")
            
        start_time = time.time()
        N_ch, N_sample = self.G.shape
        
        if self.decompMethod == 4:
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
        return ((1 - torch.abs(a)**2) ** 0.5) / (1 - torch.conj(a) * torch.exp(1j * t))
    
    def genEva(self):
        if len(self.dic_an) == 0:
            raise ValueError("Please generate the searching dictionary first!!")

        start_time = time.time()

        N_ch, N_sample = self.G.shape
        self.Base = []
        
        if self.decompMethod == 4:
            for i_ch in range(N_ch):
                dic_row, dic_col = self.dic_an[i_ch].shape
                base_ch = torch.zeros((dic_row, dic_col, N_sample), dtype=torch.complex64, device=self.device)
                
                for i in range(dic_row):
                    for j in range(dic_col):
                        if not torch.isnan(self.dic_an[i_ch][i, j]):
                            ea = self._e_a(self.dic_an[i_ch][i, j], self.t[[i_ch], :])
                            base_ch[i, j, :] = torch.fft.fft(ea, N_sample)
                            
                self.Base.append(base_ch)
        else:
            raise ValueError('This implementation only supports decompMethod=4 (Multi-channel Fast AFD)')

        self.time_genEva = time.time() - start_time
    
    def init_decomp(self):
        N_ch, _ = self.G.shape
        self.S1 = [[] for _ in range(N_ch)]
        self.max_loc = [[] for _ in range(N_ch)]
        self.an = [[] for _ in range(N_ch)]
        self.coef = [[] for _ in range(N_ch)]
        self.level = 0
        self.remainder = [[] for _ in range(N_ch)]
        for i_ch in range(N_ch):
            self.remainder[i_ch].append(self.G[[i_ch], :].clone())
        self.tem_B = [[] for _ in range(N_ch)]
        self.deComp = [[] for _ in range(N_ch)]
        self.run_time = []

        start_time = time.time()

        for i_ch in range(N_ch):
            self.S1[i_ch].append(None)
            self.max_loc[i_ch].append(None)
            an = torch.tensor(0.0, dtype=torch.complex64, device=self.device)
            self.an[i_ch].append(an)
            
            coef = self._calCoef(an, self.t[[i_ch], :], self.remainder[i_ch][self.level], self.weight)
            self.coef[i_ch].append(coef)
            
            tem_B = (torch.sqrt(1-torch.abs(an)**2) / (1-torch.conj(an) * torch.exp(self.t[[i_ch], :] * 1j)))
            self.tem_B[i_ch].append(tem_B)
            
            deComp = self.coef[i_ch][self.level] * self.tem_B[i_ch][self.level]
            self.deComp[i_ch].append(deComp)
            
            remainder = (self.remainder[i_ch][self.level] - coef * self._e_a(an, self.t[[i_ch], :])) * \
                       (1 - torch.conj(an) * torch.exp(1j * self.t[[i_ch], :])) / \
                       (torch.exp(1j * self.t[[i_ch], :]) - an)
            self.remainder[i_ch].append(remainder)

        self.run_time.append(time.time() - start_time)
    
    def _calCoef(self, a, t, G, W):
        return torch.conj(self._e_a(a, t).matmul((G.conj().T * W))) / G.shape[1]
    
    def nextDecomp(self):
        N_ch, _ = self.G.shape
        self.level += 1
        start_time = time.time()
        
        if self.decompMethod == 4:
            S1_tmp_sum = None
            
            for i_ch in range(N_ch):
                Base = self.Base[i_ch][0, :, :]
                
                fft_remainder = torch.fft.fft(self.remainder[i_ch][self.level] * self.weight.T, self.t.shape[1])
                fft_remainder_repeated = fft_remainder.repeat(Base.shape[0], 1)
                
                S1_tmp = torch.abs(torch.fft.ifft(fft_remainder_repeated * Base, self.t.shape[1], dim=1))
                S1_tmp = S1_tmp.T
                
                if S1_tmp_sum is None:
                    S1_tmp_sum = S1_tmp.clone()
                else:
                    S1_tmp_sum = S1_tmp_sum + S1_tmp
                    
            S1_tmp = S1_tmp_sum / N_ch
            
            for i_ch in range(N_ch):
                if i_ch == 0:
                    self.S1[i_ch].append(S1_tmp)
                    max_val = torch.max(S1_tmp)
                    max_indices = torch.where(S1_tmp == max_val)
                    max_row, max_col = max_indices[0][0].item(), max_indices[1][0].item()
                    self.max_loc[i_ch].append((max_row, max_col))
                    
                    dic_an = self.dic_an_search[i_ch]
                    an = dic_an[max_row, max_col]
                    self.an[i_ch].append(an)
                    
                    coef = self._calCoef(an, self.t[[i_ch], :], self.remainder[i_ch][self.level], self.weight)
                    self.coef[i_ch].append(coef)
                    
                    tem_B = (torch.sqrt(1-torch.abs(an)**2) / (1-torch.conj(an) * torch.exp(self.t[[i_ch], :] * 1j))) * \
                           ((torch.exp(1j * self.t[[i_ch], :]) - self.an[i_ch][self.level-1]) / \
                           (torch.sqrt(1 - torch.abs(self.an[i_ch][self.level-1])**2))) * \
                           self.tem_B[i_ch][self.level-1]
                    self.tem_B[i_ch].append(tem_B)
                    
                    deComp = self.coef[i_ch][self.level] * self.tem_B[i_ch][self.level]
                    self.deComp[i_ch].append(deComp)
                    
                    remainder = (self.remainder[i_ch][self.level] - coef * self._e_a(an, self.t[[i_ch], :])) * \
                               (1 - torch.conj(an) * torch.exp(1j * self.t[[i_ch], :])) / \
                               (torch.exp(1j * self.t[[i_ch], :]) - an)
                    self.remainder[i_ch].append(remainder)
                else:
                    self.S1[i_ch].append(self.S1[0][-1])
                    self.max_loc[i_ch].append(self.max_loc[0][-1])
                    self.an[i_ch].append(self.an[0][-1])
                    
                    coef = self._calCoef(self.an[0][-1], self.t[[i_ch], :], self.remainder[i_ch][self.level], self.weight)
                    self.coef[i_ch].append(coef)
                    
                    tem_B = (torch.sqrt(1-torch.abs(self.an[0][-1])**2) / (1-torch.conj(self.an[0][-1]) * torch.exp(self.t[[i_ch], :] * 1j))) * \
                           ((torch.exp(1j * self.t[[i_ch], :]) - self.an[i_ch][self.level-1]) / \
                           (torch.sqrt(1 - torch.abs(self.an[i_ch][self.level-1])**2))) * \
                           self.tem_B[i_ch][self.level-1]
                    self.tem_B[i_ch].append(tem_B)
                    
                    deComp = self.coef[i_ch][self.level] * self.tem_B[i_ch][self.level]
                    self.deComp[i_ch].append(deComp)
                    
                    remainder = (self.remainder[i_ch][self.level] - coef * self._e_a(self.an[0][-1], self.t[[i_ch], :])) * \
                               (1 - torch.conj(self.an[0][-1]) * torch.exp(1j * self.t[[i_ch], :])) / \
                               (torch.exp(1j * self.t[[i_ch], :]) - self.an[0][-1])
                    self.remainder[i_ch].append(remainder)
        else:
            raise ValueError('This implementation only supports decompMethod=4 (Multi-channel Fast AFD)')
            
        self.run_time.append(time.time() - start_time)
    
    def reconstruct(self, level):
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
        if self.level >= level:
            warnings.warn("The current decomposition already includes the given level.")
        if self.level <= 0:
            self.init_decomp()
        while self.level < level:
            self.nextDecomp()

    def decompose_batch(self, signals, n_components=8):
        batch_size = signals.shape[0]
        results = []
        
        max_batch = min(batch_size, 8)
        for b in range(0, batch_size, max_batch):
            end = min(b + max_batch, batch_size)
            mini_batch = signals[b:end]
            
            mini_batch_results = []
            for i in range(mini_batch.shape[0]):
                signal = mini_batch[i]
                
                try:
                    self.loadInputSignal(signal)
                    self.genDic(dist=0.05, max_an_mag=0.95)
                    self.genEva()
                    self.init_decomp()
                    
                    for _ in range(n_components - 1):
                        try:
                            self.nextDecomp()
                        except Exception as e:
                            print(f"Decomposition step failed: {str(e)}")
                            break
                        
                    orig_data = self.s.cpu().numpy()
                    recon = self.reconstruct(self.level).cpu().numpy()
                    
                    components = []
                    for ch in range(len(self.deComp)):
                        ch_comps = []
                        for comp in self.deComp[ch]:
                            if hasattr(comp, 'cpu'):
                                ch_comps.append(comp.cpu().detach())
                            else:
                                ch_comps.append(comp)
                        components.append(ch_comps)
                    
                    result = {
                        'original': orig_data,
                        'reconstructed': recon,
                        'components': components,
                        'parameters': {
                            'a': [self.an[ch].copy() if hasattr(self.an[ch], 'copy') else self.an[ch] 
                                 for ch in range(len(self.an))],
                            'coef': [self.coef[ch].copy() if hasattr(self.coef[ch], 'copy') else self.coef[ch] 
                                   for ch in range(len(self.coef))],
                            'level': self.level
                        }
                    }
                    
                    mini_batch_results.append(result)
                except Exception as e:
                    print(f"Batch processing failed for sample {i}: {str(e)}")
                    dummy_result = {
                        'original': signal.cpu().numpy() if hasattr(signal, 'cpu') else signal,
                        'reconstructed': signal.cpu().numpy() if hasattr(signal, 'cpu') else signal,
                        'components': [[torch.zeros_like(signal)]],
                        'parameters': {'a': [[]], 'coef': [[]], 'level': 0}
                    }
                    mini_batch_results.append(dummy_result)
                    
            results.extend(mini_batch_results)
            
        return results

    def extract_components(self, mafd_results, high_low_split_ratio=0.5):
        high_freq = []
        low_freq = []
        
        for ch_idx in range(len(mafd_results['components'])):
            components = mafd_results['components'][ch_idx]
            
            if not components or len(components) == 0:
                continue

            split_idx = max(1, int(len(components) * high_low_split_ratio))
            
            high_freq_components = []
            for comp in components[:split_idx]:
                if hasattr(comp, 'cpu'):
                    comp_np = comp.cpu().numpy()
                else:
                    comp_np = comp
                    
                if np.iscomplexobj(comp_np):
                    comp_np = np.real(comp_np)
                high_freq_components.append(comp_np)
                
            low_freq_components = []
            for comp in components[split_idx:]:
                if hasattr(comp, 'cpu'):
                    comp_np = comp.cpu().numpy()
                else:
                    comp_np = comp
                    
                if np.iscomplexobj(comp_np):
                    comp_np = np.real(comp_np)
                low_freq_components.append(comp_np)
                
            if high_freq_components:
                high_freq.append(np.sum(high_freq_components, axis=0))
            if low_freq_components:
                low_freq.append(np.sum(low_freq_components, axis=0))
        
        if not high_freq:
            high_freq = np.zeros((1, mafd_results['original'].shape[1]))
        else:
            high_freq = np.array(high_freq)
            
        if not low_freq:
            low_freq = np.zeros((1, mafd_results['original'].shape[1]))
        else:
            low_freq = np.array(low_freq)
            
        return torch.tensor(high_freq, dtype=torch.float32, device=self.device), \
               torch.tensor(low_freq, dtype=torch.float32, device=self.device)