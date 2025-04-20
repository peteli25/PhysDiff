import torch
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from sklearn.preprocessing import StandardScaler

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, time, stable, label, window_size):
        self.window_size = window_size
        
        if isinstance(data, np.ndarray):
            self.data = torch.tensor(data, dtype=torch.float32)
            self.time = torch.tensor(time, dtype=torch.float32)
            self.stable = torch.tensor(stable, dtype=torch.float32)
            self.label = torch.tensor(label, dtype=torch.float32)
        else:
            self.data = data
            self.time = time
            self.stable = stable
            self.label = label
            
        self.data_len = len(self.data)
        
    def __getitem__(self, index):
        end_idx = min(index + self.window_size, self.data_len)
        if end_idx - index < self.window_size:
            data_window = torch.zeros((self.window_size, self.data.shape[1]), dtype=self.data.dtype)
            time_window = torch.zeros((self.window_size, self.time.shape[1]), dtype=self.time.dtype)
            stable_window = torch.zeros((self.window_size, self.stable.shape[1]), dtype=self.stable.dtype)
            label_window = torch.zeros((self.window_size, self.label.shape[1]), dtype=self.label.dtype)
            
            actual_size = end_idx - index
            data_window[:actual_size] = self.data[index:end_idx]
            time_window[:actual_size] = self.time[index:end_idx]
            stable_window[:actual_size] = self.stable[index:end_idx]
            label_window[:actual_size] = self.label[index:end_idx]
        else:
            data_window = self.data[index:end_idx]
            time_window = self.time[index:end_idx]
            stable_window = self.stable[index:end_idx]
            label_window = self.label[index:end_idx]
            
        return data_window, time_window, stable_window, label_window

    def __len__(self):
        return self.data_len - self.window_size + 1


def efficient_trend_decomposition(data, window_size=1440):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    if window_size % 2 == 0:
        window_size += 1
    
    trend = np.zeros_like(data)
    for i in range(data.shape[1]):
        trend[:, i] = median_filter(data[:, i], size=min(window_size, len(data)), mode='reflect')
    
    stable = data - trend
    
    return data, trend, stable


def getTimeEmbedding(time):
    """
    Convert timestamps to cyclic time features.
    
    Parameters:
    -----------
    time : array-like
        Array of timestamps
    
    Returns:
    --------
    ndarray: Time embeddings with cyclic features
    """
    df = pd.DataFrame(time, columns=['time'])
    df['time'] = pd.to_datetime(df['time'])

    df['minute'] = df['time'].apply(lambda row: row.minute / 59 - 0.5)
    df['hour'] = df['time'].apply(lambda row: row.hour / 23 - 0.5)
    df['weekday'] = df['time'].apply(lambda row: row.weekday() / 6 - 0.5)
    df['day'] = df['time'].apply(lambda row: row.day / 30 - 0.5)
    df['month'] = df['time'].apply(lambda row: row.month / 365 - 0.5)

    return df[['minute', 'hour', 'weekday', 'day', 'month']].values


def getDiscreteChannels(dataset):
    if dataset == "MSL":
        return list(range(1, 55))
    elif dataset == "SMAP":
        return list(range(1, 25))
    elif dataset == "SWAT":
        return [2, 4, 9, 10, 11, 13, 15, 19, 20, 21, 22, 29, 30, 31, 32, 33, 42, 43, 48, 50]
    return None


def getData(path='./dataset/', dataset='SWaT', period=1440, train_rate=0.9):
    """
    Load and preprocess time series data with efficient decomposition
    
    Parameters:
    -----------
    path : str
        Path to dataset directory
    dataset : str
        Dataset name
    period : int
        Window size for trend extraction
    train_rate : float
        Ratio of training data
        
    Returns:
    --------
    dict: Processed data dictionary
    """
    init_data = np.load(path + dataset + '/' + dataset + '_train_data.npy')
    init_time = getTimeEmbedding(np.load(path + dataset + '/' + dataset + '_train_date.npy'))

    test_data = np.load(path + dataset + '/' + dataset + '_test_data.npy')
    test_time = getTimeEmbedding(np.load(path + dataset + '/' + dataset + '_test_date.npy'))
    test_label = np.load(path + dataset + '/' + dataset + '_test_label.npy')

    discrete_channels = getDiscreteChannels(dataset)
    if discrete_channels is not None:
        print(f"Removing {len(discrete_channels)} discrete channels from {dataset} dataset")
        init_data = np.delete(init_data, discrete_channels, axis=1)
        test_data = np.delete(test_data, discrete_channels, axis=1)

    scaler = StandardScaler()
    scaler.fit(init_data)
    init_data = pd.DataFrame(scaler.transform(init_data)).fillna(0).values
    test_data = pd.DataFrame(scaler.transform(test_data)).fillna(0).values

    init_data, init_trend, init_stable = efficient_trend_decomposition(init_data, window_size=period)
    init_label = np.zeros((len(init_data), 1))
    
    test_data, test_trend, test_stable = efficient_trend_decomposition(test_data, window_size=period)

    train_size = int(train_rate * len(init_data))
    
    train_data = init_data[:train_size]
    train_time = init_time[:train_size]
    train_stable = init_stable[:train_size]
    train_label = init_label[:train_size]

    valid_data = init_data[train_size:]
    valid_time = init_time[train_size:]
    valid_stable = init_stable[train_size:]
    valid_label = init_label[train_size:]

    data = {
        'train_data': train_data, 
        'train_time': train_time, 
        'train_stable': train_stable, 
        'train_label': train_label,
        'valid_data': valid_data, 
        'valid_time': valid_time, 
        'valid_stable': valid_stable, 
        'valid_label': valid_label,
        'init_data': init_data, 
        'init_time': init_time, 
        'init_stable': init_stable, 
        'init_label': init_label,
        'test_data': test_data, 
        'test_time': test_time, 
        'test_stable': test_stable, 
        'test_label': test_label
    }

    return data