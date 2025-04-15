import argparse
import torch
import numpy as np
import os
import random
from exp.solver import Solver

def setSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Physically-Guided Time Series Anomaly Detection')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='SMD', help='dataset name')
    parser.add_argument('--data_dir', type=str, default='./dataset/', help='path of data')
    parser.add_argument('--model_dir', type=str, default='./checkpoint/', help='path for checkpoints')
    
    # Training parameters
    parser.add_argument('--itr', type=int, default=1, help='number of iterations for evaluation')
    parser.add_argument('--epochs', type=int, default=10, help='training epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    
    # Time series parameters
    parser.add_argument('--period', type=int, default=1440, help='approximate period of time series')
    parser.add_argument('--train_rate', type=float, default=0.8, help='training data ratio')
    parser.add_argument('--window_size', type=int, default=64, help='sliding window size')
    
    # Model parameters
    parser.add_argument('--model_dim', type=int, default=512, help='model hidden dimension')
    parser.add_argument('--ff_dim', type=int, default=2048, help='feed-forward dimension')
    parser.add_argument('--atten_dim', type=int, default=64, help='attention dimension')
    parser.add_argument('--block_num', type=int, default=2, help='number of blocks')
    parser.add_argument('--head_num', type=int, default=8, help='number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    
    # Diffusion parameters
    parser.add_argument('--time_steps', type=int, default=1000, help='diffusion time steps')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='start value of diffusion beta')
    parser.add_argument('--beta_end', type=float, default=0.01, help='end value of diffusion beta')
    parser.add_argument('--t', type=int, default=500, help='noise step for diffusion')
    
    # MAFD parameters
    parser.add_argument('--mafd_components', type=int, default=8, help='number of MAFD components')
    parser.add_argument('--lambda_aspe', type=float, default=0.2, help='weight for ASPE loss')
    parser.add_argument('--handle_discrete', type=bool, default=True, help='whether to handle discrete channels specially')
    
    # Additional parameters
    parser.add_argument('--p', type=float, default=10.0, help='disturbance magnitude')
    parser.add_argument('--d', type=int, default=30, help='shift of period')
    parser.add_argument('--q', type=float, default=0.01, help='init anomaly probability for SPOT')
    
    # System parameters
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    
    # Parse arguments
    config = vars(parser.parse_args())
    
    # Set random seed
    setSeed(config['random_seed'])
    
    # Set GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(config['gpu_id'])
    
    # Run experiments
    results = []
    for i in range(config['itr']):
        print(f"\n===== Iteration {i+1}/{config['itr']} =====")
        exp = Solver(config)
        exp.train()
        result = exp.test()
        results.append(result)
    
    # Calculate average metrics
    avg_precision = np.mean([r['precision'] for r in results])
    avg_recall = np.mean([r['recall'] for r in results])
    avg_f1 = np.mean([r['f1_score'] for r in results])
    
    print("\n===== Final Results =====")
    print(f"Dataset: {config['dataset']}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print("==========================")