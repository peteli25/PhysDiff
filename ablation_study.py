import os
import argparse
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import yaml
import logging
from datetime import datetime

from exp.solver import Solver
from utils.seed import setSeed
from utils.evaluate import evaluate
from models.detector import PhysicallyGuidedDiffusion
from utils.earlystop import EarlyStop
from data.preprocess import getData
from data.dataset import Dataset
from torch.utils.data import DataLoader


class AblationPhysicallyGuidedDiffusion(PhysicallyGuidedDiffusion):
    def __init__(self, time_steps=1000, beta_start=0.0001, beta_end=0.02, 
                 window_size=64, model_dim=512, ff_dim=2048, atten_dim=64,
                 feature_num=8, time_num=5, block_num=2, head_num=8,
                 dropout=0.1, device=None, d=30, t=500, mafd_components=8,
                 use_physics_extraction=True, use_aspe=True, use_routing_attention=True,
                 use_high_freq=True, use_low_freq=True, use_physical_guidance=True,
                 use_dynamic_weights=True, use_physical_loss=True):
        super().__init__(time_steps, beta_start, beta_end, window_size, model_dim, 
                         ff_dim, atten_dim, feature_num, time_num, block_num, 
                         head_num, dropout, device, d, t, mafd_components)
        
        self.use_physics_extraction = use_physics_extraction
        self.use_aspe = use_aspe
        self.use_routing_attention = use_routing_attention
        self.use_high_freq = use_high_freq
        self.use_low_freq = use_low_freq
        self.use_physical_guidance = use_physical_guidance
        self.use_dynamic_weights = use_dynamic_weights
        self.use_physical_loss = use_physical_loss
    
    def forward(self, data, time, p=0):
        batch_size, window_size, feature_num = data.shape
        
        if p > 0:
            disturb = torch.rand(batch_size, feature_num, device=self.device) * p
            disturb = disturb.unsqueeze(1).repeat(1, window_size, 1)
            data_disturbed = data + disturb
        else:
            data_disturbed = data
            disturb = 0
        
        subset_size = min(batch_size, 4)
        subset_indices = torch.randperm(batch_size)[:subset_size]
        subset_data = data_disturbed[subset_indices]
        
        try:
            if self.use_physics_extraction:
                ph_high_subset, ph_low_subset, complexity = self.feature_extractor(subset_data)
            else:
                ph_high_subset, ph_low_subset, complexity = self._simple_spectral_extraction(subset_data)
            
            if not self.use_aspe:
                complexity = 0.5
            
            self.freq_memory.update(ph_high_subset, ph_low_subset, complexity)
            ph_high, ph_low, complexity = self.freq_memory.sample(batch_size)
            
            if ph_high is None:
                if len(ph_high_subset.shape) == 2: 
                    ph_high_subset = ph_high_subset.unsqueeze(0)
                if len(ph_low_subset.shape) == 2: 
                    ph_low_subset = ph_low_subset.unsqueeze(0)
                    
                ph_high = ph_high_subset.expand(batch_size, -1, -1) if ph_high_subset.shape[0] == 1 else ph_high_subset
                ph_low = ph_low_subset.expand(batch_size, -1, -1) if ph_low_subset.shape[0] == 1 else ph_low_subset
        except Exception as e:
            print(f"Feature extraction failed: {str(e)}")
            ph_high = torch.zeros((batch_size, feature_num, window_size), device=self.device)
            ph_low = torch.zeros((batch_size, feature_num, window_size), device=self.device)
            complexity = 0.5
        
        if not self.use_high_freq:
            ph_high = torch.zeros_like(ph_high)
        if not self.use_low_freq:
            ph_low = torch.zeros_like(ph_low)
            
        if ph_high.dim() > 3:
            ph_high = ph_high.view(batch_size, -1, window_size)
        if ph_low.dim() > 3:
            ph_low = ph_low.view(batch_size, -1, window_size)
        
        ph_high_emb = self.high_freq_proj(ph_high.unsqueeze(-1))
        ph_low_emb = self.low_freq_proj(ph_low.unsqueeze(-1))
        
        bt = torch.full((batch_size,), self.t, device=self.device)
        sample_noise = torch.randn_like(data_disturbed)
        
        try:
            if self.use_physical_guidance:
                noise_data = self.diffusion.q_sample(data_disturbed, ph_low, bt, sample_noise, high_freq=ph_high)
            else:
                noise_data = self._vanilla_q_sample(data_disturbed, bt, sample_noise)
        except Exception as e:
            print(f"Diffusion sampling failed: {str(e)}")
            noise_data = data_disturbed
        
        data_emb = self.data_embedding(noise_data) + self.position_embedding(noise_data)
        time_emb = self.time_embedding(time)
        x = data_emb + time_emb
        
        for block in self.encoder_blocks:
            if self.use_routing_attention:
                x = block(x, ph_high_emb, ph_low_emb)
            else:
                x = self._vanilla_attention_block(block, x)
            
        recon = self.reconstruction_head(x)
        score = self.anomaly_predictor(x)
        
        if isinstance(disturb, torch.Tensor):
            recon = recon - disturb
            
        return recon, score, complexity
    
    def _simple_spectral_extraction(self, x):
        batch_size, seq_len, features = x.shape
        
        x_reshaped = x.permute(0, 2, 1)
        
        x_fft = torch.fft.rfft(x_reshaped, dim=2)
        
        fft_len = x_fft.shape[2]
        split_idx = fft_len // 2
        
        high_freq_fft = torch.zeros_like(x_fft)
        high_freq_fft[:, :, split_idx:] = x_fft[:, :, split_idx:]
        
        low_freq_fft = torch.zeros_like(x_fft)
        low_freq_fft[:, :, :split_idx] = x_fft[:, :, :split_idx]
        
        high_freq = torch.fft.irfft(high_freq_fft, n=seq_len, dim=2)
        low_freq = torch.fft.irfft(low_freq_fft, n=seq_len, dim=2)
        
        complexity = 0.5
        
        return high_freq, low_freq, complexity
    
    def _vanilla_q_sample(self, x_start, batch_t, noise):
        sqrt_alphas_cumprod_t = self.diffusion._extract(self.diffusion.sqrt_alphas_cumprod, batch_t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.diffusion._extract(self.diffusion.sqrt_one_minus_alphas_cumprod, batch_t, x_start.shape)
        
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_noisy
    
    def _vanilla_attention_block(self, block, x):
        residual = x.clone()
        x = block.time_block(x) if hasattr(block, 'time_block') else x
        
        if hasattr(block, 'feature_block'):
            x = block.feature_block(x)
        
        return x
    
    def compute_loss(self, data, time, stable=None, label=None, p=0, lambda_aspe=0.1):
        recon, score, complexity = self.forward(data, time, p)
        
        recon_loss = torch.nn.functional.mse_loss(recon, data)
        
        if self.use_physical_loss and self.use_aspe:
            aspe_loss = complexity
        else:
            aspe_loss = torch.tensor(0.0, device=self.device)
        
        if label is not None:
            pred = torch.sigmoid(score)
            gamma = 2.0
            alpha = 0.25
            pt = label * pred + (1 - label) * (1 - pred)
            focal_weight = alpha * torch.pow(1 - pt, gamma)
            detection_loss = -torch.mean(focal_weight * (label * torch.log(pred + 1e-10) + 
                                                    (1 - label) * torch.log(1 - pred + 1e-10)))
        else:
            detection_loss = torch.tensor(0.0, device=self.device)
        
        if self.use_physical_loss:
            total_loss = recon_loss + lambda_aspe * aspe_loss
            if label is not None:
                total_loss += detection_loss
        else:
            total_loss = recon_loss
            if label is not None:
                total_loss += detection_loss
                
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'aspe_loss': aspe_loss,
            'detection_loss': detection_loss
        }


class AblationSolver(Solver):
    def __init__(self, config):
        # Initialize with the config directly to avoid calling _get_data() prematurely
        self.__dict__.update(config)
        
        # Create necessary directories
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'\nDevice: {self.device}')
        
        # Now call data loading explicitly
        self._get_data()
        
        # Initialize ablation model
        self._init_ablation_model()
        
        # Initialize training state
        self.current_epoch = 0
    
    def _init_ablation_model(self):
        self.model = AblationPhysicallyGuidedDiffusion(
            time_steps=self.time_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            window_size=self.window_size,
            model_dim=self.model_dim,
            ff_dim=self.ff_dim,
            atten_dim=self.atten_dim,
            feature_num=self.feature_num,
            time_num=self.time_num,
            block_num=self.block_num,
            head_num=self.head_num,
            dropout=self.dropout,
            device=self.device,
            d=self.d,
            t=self.t,
            mafd_components=getattr(self, 'mafd_components', 8),
            use_physics_extraction=getattr(self, 'use_physics_extraction', True),
            use_aspe=getattr(self, 'use_aspe', True),
            use_routing_attention=getattr(self, 'use_routing_attention', True),
            use_high_freq=getattr(self, 'use_high_freq', True),
            use_low_freq=getattr(self, 'use_low_freq', True),
            use_physical_guidance=getattr(self, 'use_physical_guidance', True),
            use_dynamic_weights=getattr(self, 'use_dynamic_weights', True),
            use_physical_loss=getattr(self, 'use_physical_loss', True)
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2, eta_min=self.lr * 0.01
        )
        
        self.early_stopping = EarlyStop(
            patience=self.patience, 
            path=os.path.join(self.model_dir, f"{self.dataset}_ablation_model.pkl")
        )


def run_ablation_experiment(config, ablation_name, ablation_config=None):
    print(f"\n===== Running Ablation Experiment: {ablation_name} =====")
    
    # Create a deep copy of the config to avoid modifying the original
    exp_config = config.copy()
    
    # Update with ablation-specific configuration
    if ablation_config:
        exp_config.update(ablation_config)
        
    # Set random seed for reproducibility
    setSeed(exp_config['random_seed'])
    
    # Create and train the model
    try:
        exp = AblationSolver(exp_config)
        
        # Train the model
        exp.train()
        
        # Test the model
        result = exp.test()
        
        # Print results
        print(f"Ablation: {ablation_name}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall: {result['recall']:.4f}")
        print(f"F1 Score: {result['f1_score']:.4f}")
        
        # Return the results
        return {
            'ablation': ablation_name,
            'precision': float(result['precision']),
            'recall': float(result['recall']),
            'f1_score': float(result['f1_score']),
            'threshold': float(result['threshold']) if 'threshold' in result else None
        }
    except Exception as e:
        print(f"Error in experiment {ablation_name}: {str(e)}")
        # Return empty result with error
        return {
            'ablation': ablation_name,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'error': str(e)
        }


def run_ablation_study(dataset='SMD', data_dir='./dataset/', model_dir='./checkpoint/',
                      result_dir='./ablation_results/'):
    # Set up base configuration
    config = {
        'dataset': dataset,
        'data_dir': data_dir,
        'model_dir': model_dir,
        'epochs': 10,
        'patience': 3,
        'batch_size': 64,
        'lr': 0.0001,
        'period': 1440,  # Critical parameter for time series data
        'train_rate': 0.8,
        'window_size': 64,
        'model_dim': 512,
        'ff_dim': 2048,
        'atten_dim': 64,
        'block_num': 2,
        'head_num': 8,
        'dropout': 0.2,
        'time_steps': 1000,
        'beta_start': 0.0001,
        'beta_end': 0.01,
        't': 500,
        'mafd_components': 4,
        'lambda_aspe': 0.1,
        'p': 10.0,
        'd': 30,
        'q': 0.01,
        'random_seed': 42,
        'gpu_id': 0
    }
    
    # Create result directories
    os.makedirs(result_dir, exist_ok=True)
    dataset_result_dir = os.path.join(result_dir, dataset)
    os.makedirs(dataset_result_dir, exist_ok=True)
    
    # Define ablation configurations
    ablation_configs = [
        {
            'name': 'Baseline (Full Model)',
            'config': {}
        },
        {
            'name': 'Remove Physics Extraction',
            'config': {'use_physics_extraction': False}
        },
        {
            'name': 'Remove ASPE',
            'config': {'use_aspe': False}
        },
        {
            'name': 'Remove Routing Attention',
            'config': {'use_routing_attention': False}
        },
        {
            'name': 'Remove High Frequency',
            'config': {'use_high_freq': False}
        },
        {
            'name': 'Remove Low Frequency',
            'config': {'use_low_freq': False}
        },
        {
            'name': 'Remove Physical Guidance',
            'config': {'use_physical_guidance': False}
        },
        {
            'name': 'Remove Dynamic Weights',
            'config': {'use_dynamic_weights': False}
        },
        {
            'name': 'Remove Physical Loss',
            'config': {'use_physical_loss': False}
        },
        {
            'name': 'Fewer MAFD Components (4)',
            'config': {'mafd_components': 4}
        },
        {
            'name': 'More MAFD Components (16)',
            'config': {'mafd_components': 16}
        }
    ]
    
    # Run ablation experiments
    results = []
    for ablation in tqdm(ablation_configs, desc=f"Running {dataset} Ablation Experiments"):
        result = run_ablation_experiment(config, ablation['name'], ablation['config'])
        results.append(result)
        
        # Save individual result
        result_file = os.path.join(dataset_result_dir, f"{ablation['name'].replace(' ', '_')}.json")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=4)
    
    # Save all results
    all_results_file = os.path.join(dataset_result_dir, 'all_results.json')
    with open(all_results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Visualize results
    visualize_ablation_results(results, dataset, dataset_result_dir)
    
    return results


def visualize_ablation_results(results, dataset_name, output_dir):
    # Extract data for visualization
    names = [r['ablation'] for r in results]
    f1_scores = [r['f1_score'] for r in results]
    precision = [r['precision'] for r in results]
    recall = [r['recall'] for r in results]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create bar chart
    x = np.arange(len(names))
    width = 0.25
    
    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x, f1_scores, width, label='F1 Score')
    ax.bar(x + width, recall, width, label='Recall')
    
    # Add labels and title
    ax.set_xlabel('Ablation Experiment')
    ax.set_ylabel('Score')
    ax.set_title(f'Ablation Experiment Results on {dataset_name} Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Add data labels
    for i, v in enumerate(f1_scores):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontweight='bold')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_plot.png'), dpi=300)
    plt.close()
    
    # Create component impact visualization if there are multiple results
    if len(results) > 1:
        baseline_f1 = results[0]['f1_score']
        relative_impact = [(r['f1_score'] - baseline_f1) / baseline_f1 * 100 for r in results[1:]]
        ablation_names = [r['ablation'] for r in results[1:]]
        
        # Sort by impact
        sorted_indices = np.argsort(relative_impact)
        sorted_impact = [relative_impact[i] for i in sorted_indices]
        sorted_names = [ablation_names[i] for i in sorted_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create horizontal bar chart
        bars = ax.barh(np.arange(len(sorted_impact)), sorted_impact, height=0.6)
        
        # Color bars by impact
        for i, bar in enumerate(bars):
            if sorted_impact[i] < 0:
                bar.set_color('red')
            else:
                bar.set_color('green')
        
        # Add labels
        ax.set_yticks(np.arange(len(sorted_impact)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Relative Impact on F1 Score (%)')
        ax.set_title(f'Component Impact Analysis for {dataset_name}')
        
        # Add reference line and grid
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add data labels
        for i, v in enumerate(sorted_impact):
            ax.text(v - 1 if v < 0 else v + 1, i, f"{v:.2f}%", va='center', fontweight='bold')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'component_impact.png'), dpi=300)
        plt.close()


def visualize_cross_dataset_results(all_results, output_dir):
    datasets = list(all_results.keys())
    ablation_names = [r['ablation'] for r in all_results[datasets[0]]]
    
    # Prepare F1 scores for each dataset
    f1_scores = {}
    for dataset in datasets:
        f1_scores[dataset] = [r['f1_score'] for r in all_results[dataset]]
    
    # Calculate average F1 scores
    avg_f1 = []
    for i in range(len(ablation_names)):
        avg = np.mean([f1_scores[dataset][i] for dataset in datasets])
        avg_f1.append(avg)
    
    # 1. F1 Score Heatmap
    plt.figure(figsize=(14, 10))
    
    heatmap_data = np.zeros((len(datasets), len(ablation_names)))
    for i, dataset in enumerate(datasets):
        for j in range(len(ablation_names)):
            heatmap_data[i, j] = f1_scores[dataset][j]
    
    plt.imshow(heatmap_data, cmap='viridis')
    plt.colorbar(label='F1 Score')
    
    plt.yticks(np.arange(len(datasets)), datasets)
    plt.xticks(np.arange(len(ablation_names)), ablation_names, rotation=45, ha='right')
    
    plt.title('F1 Score Heatmap Across Datasets')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_f1_scores.png'), dpi=300)
    plt.close()
    
    # 2. Average F1 Score Bar Chart
    plt.figure(figsize=(14, 8))
    
    baseline_f1 = avg_f1[0]
    
    relative_impact = [(score - baseline_f1) / baseline_f1 * 100 for score in avg_f1]
    
    colors = ['green' if x >= 0 else 'red' for x in relative_impact]
    
    plt.bar(ablation_names, relative_impact, color=colors)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.ylabel('Relative F1 Score Change (%)')
    plt.title('Average Component Impact Across All Datasets')
    plt.xticks(rotation=45, ha='right')
    
    for i, v in enumerate(relative_impact):
        plt.text(i, v + (0.5 if v >= 0 else -1.5), 
                 f"{v:.2f}%", 
                 ha='center', 
                 fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'component_impact.png'), dpi=300)
    plt.close()
    
    # 3. Dataset Comparison
    plt.figure(figsize=(14, 8))
    
    best_f1 = {}
    worst_f1 = {}
    baseline_f1 = {}
    
    for dataset in datasets:
        scores = f1_scores[dataset]
        best_f1[dataset] = max(scores)
        worst_f1[dataset] = min(scores)
        baseline_f1[dataset] = scores[0]
    
    x = np.arange(len(datasets))
    width = 0.25
    
    plt.bar(x - width, [baseline_f1[d] for d in datasets], width, label='Baseline Model')
    plt.bar(x, [best_f1[d] for d in datasets], width, label='Best Configuration')
    plt.bar(x + width, [worst_f1[d] for d in datasets], width, label='Worst Configuration')
    
    plt.xlabel('Dataset')
    plt.ylabel('F1 Score')
    plt.title('Model Configuration Performance Across Datasets')
    plt.xticks(x, datasets)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_comparison.png'), dpi=300)
    plt.close()


def setup_logger(log_dir='./logs/'):
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"ablation_experiment_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('PhysDiff_Ablation')


def main():
    parser = argparse.ArgumentParser(description='PhysDiff Ablation Experiments')
    
    # Basic parameters
    parser.add_argument('--data_dir', type=str, default='./dataset/', help='Data directory')
    parser.add_argument('--model_dir', type=str, default='./checkpoint/', help='Model directory')
    parser.add_argument('--result_dir', type=str, default='./results/', help='Results directory')
    parser.add_argument('--log_dir', type=str, default='./logs/', help='Log directory')
    parser.add_argument('--datasets', type=str, nargs='+', default=['SWaT'], 
                        help='Datasets to test')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    
    # Time series parameters
    parser.add_argument('--period', type=int, default=1440, help='Approximate period of time series')
    
    # System parameters
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    parser.add_argument('--skip_existing', action='store_true', help='Skip existing experiments')
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger(args.log_dir)
    logger.info(f"Starting PhysDiff ablation experiments - Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Parameters: {vars(args)}")
    
    # Set random seed
    setSeed(args.random_seed)
    logger.info(f"Random seed set to: {args.random_seed}")
    
    # Setup GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        logger.info(f"Using GPU: {args.gpu_id} ({torch.cuda.get_device_name(args.gpu_id)})")
    else:
        logger.warning("No GPU detected, using CPU for experiments")
    
    # Create result directory
    os.makedirs(args.result_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {args.result_dir}")
    
    # Results for all datasets
    all_datasets_results = {}
    
    # Run ablation study for each dataset
    start_time = time.time()
    for dataset_idx, dataset in enumerate(args.datasets):
        logger.info(f"\n{'='*20} Starting {dataset} Dataset ({dataset_idx+1}/{len(args.datasets)}) {'='*20}")
        
        # Create dataset result directory
        dataset_result_dir = os.path.join(args.result_dir, dataset)
        os.makedirs(dataset_result_dir, exist_ok=True)
        
        # Run ablation study
        dataset_config = {
            'dataset': dataset,
            'data_dir': args.data_dir,
            'model_dir': args.model_dir,
            'result_dir': os.path.join(args.result_dir, dataset),
            'epochs': args.epochs,
            'patience': args.patience,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'period': args.period,  # Critical parameter
            'random_seed': args.random_seed,
            'gpu_id': args.gpu_id
        }
        
        # Check if results already exist
        result_file = os.path.join(dataset_result_dir, 'all_results.json')
        if args.skip_existing and os.path.exists(result_file):
            logger.info(f"Found existing results file {result_file}, skipping...")
            with open(result_file, 'r') as f:
                dataset_results = json.load(f)
        else:
            # Set up the base config for this dataset
            base_config = {
                'dataset': dataset,
                'data_dir': args.data_dir,
                'model_dir': args.model_dir,
                'epochs': args.epochs,
                'patience': args.patience,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'period': args.period,  # Critical parameter
                'train_rate': 0.8,
                'window_size': 64,
                'model_dim': 512,
                'ff_dim': 2048,
                'atten_dim': 64,
                'block_num': 2,
                'head_num': 8,
                'dropout': 0.2,
                'time_steps': 1000,
                'beta_start': 0.0001,
                'beta_end': 0.01,
                't': 500,
                'mafd_components': 4,
                'lambda_aspe': 0.1,
                'p': 10.0,
                'd': 30,
                'q': 0.01,
                'random_seed': args.random_seed,
                'gpu_id': args.gpu_id
            }
            
            # Run ablation experiments
            logger.info(f"Running ablation experiments for {dataset}")
            dataset_results = run_ablation_study(
                dataset=dataset, 
                data_dir=args.data_dir, 
                model_dir=args.model_dir,
                result_dir=args.result_dir
            )
            
        # Add to all results
        all_datasets_results[dataset] = dataset_results
        
    # Calculate total runtime
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"All dataset experiments completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Save all results
    all_results_file = os.path.join(args.result_dir, 'all_datasets_results.json')
    with open(all_results_file, 'w') as f:
        json.dump(all_datasets_results, f, indent=4)
    logger.info(f"All results saved to {all_results_file}")
    
    # Generate cross-dataset visualizations
    try:
        visualize_cross_dataset_results(all_datasets_results, args.result_dir)
        logger.info("Cross-dataset visualizations generated")
    except Exception as e:
        logger.error(f"Error generating cross-dataset visualizations: {str(e)}")
    
    logger.info(f"\n{'='*20} Ablation Experiments Completed {'='*20}")


if __name__ == '__main__':
    main()