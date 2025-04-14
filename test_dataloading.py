import numpy as np
import os
import sys
from data.preprocess import getData
import torch
from data.dataset import Dataset


def verify_dataset(dataset_name, window_size=100):
    """
    验证数据集是否能被原有代码正确加载
    
    参数:
    dataset_name: 数据集名称
    window_size: 滑动窗口大小
    """
    print(f"\n验证数据集 {dataset_name} 的加载:")
    
    try:
        # 使用原有的getData函数加载数据
        data = getData(dataset=dataset_name)
        
        # 打印数据集的基本信息
        print(f"训练数据形状: {data['train_data'].shape}")
        print(f"训练时间形状: {data['train_time'].shape}")
        print(f"测试数据形状: {data['test_data'].shape}")
        print(f"测试时间形状: {data['test_time'].shape}")
        print(f"测试标签形状: {data['test_label'].shape}")
        
        # 创建PyTorch数据集
        train_dataset = Dataset(
            data=torch.FloatTensor(data['train_data']),
            time=torch.FloatTensor(data['train_time']),
            stable=torch.FloatTensor(data['train_stable']),
            label=torch.FloatTensor(data['train_label']),
            window_size=window_size
        )
        
        test_dataset = Dataset(
            data=torch.FloatTensor(data['test_data']),
            time=torch.FloatTensor(data['test_time']),
            stable=torch.FloatTensor(data['test_stable']),
            label=torch.FloatTensor(data['test_label']),
            window_size=window_size
        )
        
        print(f"训练数据集大小: {len(train_dataset)}")
        print(f"测试数据集大小: {len(test_dataset)}")
        
        # 获取一个样本
        sample_data, sample_time, sample_stable, sample_label = train_dataset[0]
        print(f"样本窗口形状: {sample_data.shape}")
        
        print(f"数据集 {dataset_name} 验证成功!")
        return True
        
    except Exception as e:
        print(f"数据集 {dataset_name} 验证失败: {e}")
        return False


def main():
    # 验证原有数据集
    original_dataset = "SWaT"
    print(f"首先验证原有数据集 {original_dataset}...")
    verify_dataset(original_dataset)
    
    # 验证转换后的数据集
    new_dataset = "MSL"
    print(f"\n然后验证转换后的数据集 {new_dataset}...")
    verify_dataset(new_dataset)


if __name__ == "__main__":
    main()