import numpy as np
import os
import pandas as pd
import shutil
from datetime import datetime, timedelta
import sys


def convert_dataset_to_standard_format(source_dir, target_dir, dataset_name):
    """
    将各种数据集转换为标准格式
    
    参数:
    source_dir: 源数据集目录
    target_dir: 目标数据集目录
    dataset_name: 数据集名称
    """
    print(f"开始转换数据集: {dataset_name}")
    
    # 创建目标目录
    os.makedirs(os.path.join(target_dir, dataset_name), exist_ok=True)
    
    # 定义源文件和目标文件的路径
    source_files = {
        'train': os.path.join(source_dir, f'{dataset_name}_train.npy'),
        'test': os.path.join(source_dir, f'{dataset_name}_test.npy'),
        'test_label': os.path.join(source_dir, f'{dataset_name}_test_label.npy')
    }
    
    # 检查源文件是否存在
    for key, file_path in source_files.items():
        if not os.path.exists(file_path):
            print(f"错误: 源文件不存在: {file_path}")
            return False
    
    target_files = {
        'train_data': os.path.join(target_dir, dataset_name, f'{dataset_name}_train_data.npy'),
        'train_date': os.path.join(target_dir, dataset_name, f'{dataset_name}_train_date.npy'),
        'test_data': os.path.join(target_dir, dataset_name, f'{dataset_name}_test_data.npy'),
        'test_date': os.path.join(target_dir, dataset_name, f'{dataset_name}_test_date.npy'),
        'test_label': os.path.join(target_dir, dataset_name, f'{dataset_name}_test_label.npy')
    }
    
    # 复制并重命名数据文件
    print("复制并重命名数据文件...")
    shutil.copy(source_files['train'], target_files['train_data'])
    shutil.copy(source_files['test'], target_files['test_data'])
    
    # 读取训练和测试数据，以获取数据形状
    train_data = np.load(source_files['train'])
    test_data = np.load(source_files['test'])
    print(f"训练数据形状: {train_data.shape}")
    print(f"测试数据形状: {test_data.shape}")
    
    # 读取、处理并保存test_label文件
    print("处理测试数据标签...")
    test_label = np.load(source_files['test_label'])
    print(f"原始测试标签形状: {test_label.shape}, 类型: {test_label.dtype}")
    
    # 确保test_label是2D数组
    if len(test_label.shape) == 1:
        print(f"将1D测试标签转换为2D: {test_label.shape} -> {test_label.reshape(-1, 1).shape}")
        test_label = test_label.reshape(-1, 1)
    elif len(test_label.shape) > 2:
        print(f"警告: 测试标签维度过高 ({len(test_label.shape)}D), 转换为2D")
        # 对于高维数组，保留第一列作为标签
        test_label = test_label.reshape(test_label.shape[0], -1)[:, 0].reshape(-1, 1)
    
    # 确保第二维是1
    if test_label.shape[1] != 1:
        print(f"调整测试标签形状: {test_label.shape} -> {test_label[:, 0].reshape(-1, 1).shape}")
        test_label = test_label[:, 0].reshape(-1, 1)
    
    # 检查测试标签长度是否与测试数据匹配
    if len(test_label) != len(test_data):
        print(f"警告: 测试标签长度 ({len(test_label)}) 与测试数据长度 ({len(test_data)}) 不匹配!")
        if len(test_label) < len(test_data):
            # 如果标签少于数据，用零填充
            padding = np.zeros((len(test_data) - len(test_label), 1))
            test_label = np.vstack((test_label, padding))
            print(f"已填充标签至 {test_label.shape}")
        else:
            # 如果标签多于数据，截断
            test_label = test_label[:len(test_data)]
            print(f"已截断标签至 {test_label.shape}")
    
    # 保存处理后的test_label
    np.save(target_files['test_label'], test_label)
    print(f"已保存测试标签, 形状: {test_label.shape}")
    
    # 生成日期文件
    print("生成日期文件...")
    
    train_samples = train_data.shape[0]
    test_samples = test_data.shape[0]
    
    # 生成合成时间戳 (以1分钟为间隔)
    def generate_timestamps(num_samples, start_date=datetime(2020, 1, 1)):
        timestamps = []
        for i in range(num_samples):
            timestamp = start_date + timedelta(minutes=i)
            timestamps.append(timestamp.strftime('%Y-%m-%d %H:%M:%S'))
        return np.array(timestamps).reshape(-1, 1)
    
    # 训练数据的时间戳
    train_dates = generate_timestamps(train_samples)
    
    # 测试数据的时间戳 (接续训练数据的时间)
    test_start_date = datetime(2020, 1, 1) + timedelta(minutes=train_samples)
    test_dates = generate_timestamps(test_samples, test_start_date)
    
    # 保存时间戳文件
    np.save(target_files['train_date'], train_dates)
    np.save(target_files['test_date'], test_dates)
    
    print(f"数据集 {dataset_name} 已转换完成!")
    print(f"生成的文件保存在: {os.path.join(target_dir, dataset_name)}")
    print("文件列表:")
    for file_type, file_path in target_files.items():
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # 转换为MB
        print(f"  - {os.path.basename(file_path)}: {file_size:.2f} MB")
    
    # 验证生成的文件
    print("\n验证生成的文件:")
    for file_type, file_path in target_files.items():
        data = np.load(file_path, allow_pickle=True)
        print(f"  - {os.path.basename(file_path)}: 形状 {data.shape}, 类型 {data.dtype}")
    
    # 使用数据加载脚本验证
    print("\n尝试通过预处理加载数据...")
    try:
        # 导入getData函数进行测试 - 相对路径可能需要调整
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data.preprocess import getData
        
        data = getData(dataset=dataset_name)
        print("通过预处理加载数据成功!")
        print(f"最终数据形状:")
        print(f"  - 训练数据: {data['train_data'].shape}")
        print(f"  - 训练时间: {data['train_time'].shape}")
        print(f"  - 测试数据: {data['test_data'].shape}")
        print(f"  - 测试时间: {data['test_time'].shape}")
        print(f"  - 测试标签: {data['test_label'].shape}")
    except Exception as e:
        print(f"通过预处理加载数据失败: {e}")
    
    return True


def main():
    # 处理命令行参数
    if len(sys.argv) < 4:
        print("用法: python convert.py <源目录> <目标目录> <数据集名称>")
        print("示例: python convert.py ./raw_data ./ MSL")
        return
    
    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    dataset_name = sys.argv[3]
    
    # 转换指定数据集
    convert_dataset_to_standard_format(source_dir, target_dir, dataset_name)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # 默认配置 - 如果没有命令行参数
        source_dir = "/home/pete/下载/drive-download-20250409T142030Z-001/NIPS_TS_Swan"  # 原始数据集目录
        target_dir = "./"  # 目标目录
        dataset_name = "NIPS_TS_Swan"  # 数据集名称
        
        # 提示用户确认
        print(f"将使用默认配置:")
        print(f"  - 源目录: {source_dir}")
        print(f"  - 目标目录: {target_dir}")
        print(f"  - 数据集名称: {dataset_name}")
        confirm = input("确认继续? (y/n): ")
        
        if confirm.lower() == 'y':
            convert_dataset_to_standard_format(source_dir, target_dir, dataset_name)
        else:
            print("已取消。请使用命令行参数指定配置。")
            print("用法: python convert.py <源目录> <目标目录> <数据集名称>")