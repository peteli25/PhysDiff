import argparse
import torch
import numpy as np
import os
import json
from tqdm import tqdm

from exp.solver import Solver
from utils.visualization import visualize_model_outputs
from utils.seed import setSeed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Physically-Guided Time Series Anomaly Detection')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='SMD', help='dataset name')
    parser.add_argument('--data_dir', type=str, default='./dataset/', help='path of data')
    parser.add_argument('--model_dir', type=str, default='./checkpoint/', help='path for checkpoints')
    parser.add_argument('--result_dir', type=str, default='./results/', help='path for results')
    
    # Model parameters
    parser.add_argument('--window_size', type=int, default=64, help='sliding window size')
    parser.add_argument('--model_dim', type=int, default=512, help='model hidden dimension')
    parser.add_argument('--ff_dim', type=int, default=2048, help='feed-forward dimension')
    parser.add_argument('--atten_dim', type=int, default=64, help='attention dimension')
    parser.add_argument('--block_num', type=int, default=2, help='number of blocks')
    parser.add_argument('--head_num', type=int, default=8, help='number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    
    # Diffusion parameters
    parser.add_argument('--time_steps', type=int, default=1000, help='diffusion time steps')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='start value of diffusion beta')
    parser.add_argument('--beta_end', type=float, default=0.02, help='end value of diffusion beta')
    parser.add_argument('--t', type=int, default=500, help='noise step for diffusion')
    
    # MAFD parameters
    parser.add_argument('--mafd_components', type=int, default=4, help='number of MAFD components')
    
    # Testing parameters
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--q', type=float, default=0.01, help='anomaly probability threshold')
    parser.add_argument('--d', type=int, default=30, help='shift of period')
    parser.add_argument('--period', type=int, default=1440, help='approximate period')
    parser.add_argument('--train_rate', type=float, default=0.8, help='training ratio')
    parser.add_argument('--visualize', action='store_true', help='visualize results')
    parser.add_argument('--time_num', type=int, default=5, help='time features dimension')
    parser.add_argument('--feature_num', type=int, default=38, help='feature dimension')
    
    # System parameters
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    
    # Parse arguments
    args = parser.parse_args()
    config = vars(args)
    
    # Set random seed
    setSeed(config['random_seed'])
    
    # Set GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(config['gpu_id'])
        device = torch.device(f'cuda:{config["gpu_id"]}')
    else:
        device = torch.device('cpu')
    
    # Create result directory
    os.makedirs(config['result_dir'], exist_ok=True)
    result_path = os.path.join(config['result_dir'], config['dataset'])
    os.makedirs(result_path, exist_ok=True)
    
    print(f"Testing model on {config['dataset']} dataset...")
    
    # Load model and data
    exp = Solver(config)
    
    # Test model
    print("Running test...")
    results = exp.test()
    
    # Save results
    result_file = os.path.join(result_path, 'results.json')
    with open(result_file, 'w') as f:
        json.dump({
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1_score']),
            'threshold': float(results['threshold']),
            'config': config
        }, f, indent=4)
    
    print(f"Results saved to {result_file}")
    
    # Visualize results if requested
    if config['visualize']:
        print("Generating visualizations...")
        visualize_model_outputs(
            model=exp.model,
            dataset=exp.test_set,
            device=device,
            output_dir=os.path.join(result_path, 'visualizations'),
            max_samples=5
        )
        
    print("Done!")