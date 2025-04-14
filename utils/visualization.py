import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.gridspec import GridSpec
import os

def plot_decomposition_results(time_idx, original, reconstructed, components, save_path=None, title=None):
    """
    Plot decomposition results
    
    Parameters:
    -----------
    time_idx : array-like
        Time indices for x-axis
    original : array-like
        Original signal
    reconstructed : array-like
        Reconstructed signal
    components : list of array-like
        List of signal components
    save_path : str, optional
        Path to save the figure
    title : str, optional
        Figure title
    """
    n_components = len(components)
    fig = plt.figure(figsize=(12, 8))
    
    # Create GridSpec
    heights = [2] + [1] * n_components
    gs = GridSpec(1 + n_components, 1, height_ratios=heights)
    
    # Plot original vs reconstructed
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time_idx, original, 'b-', label='Original')
    ax1.plot(time_idx, reconstructed, 'r-', label='Reconstructed')
    ax1.legend(loc='upper right')
    ax1.set_title('Original vs Reconstructed Signal')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot individual components
    for i in range(n_components):
        ax = fig.add_subplot(gs[i+1])
        ax.plot(time_idx, components[i], 'g-')
        ax.set_title(f'Component {i+1}')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_anomaly_detection_results(time_idx, data, scores, labels=None, predictions=None, 
                                 threshold=None, save_path=None, title=None):
    """
    Plot anomaly detection results
    
    Parameters:
    -----------
    time_idx : array-like
        Time indices for x-axis
    data : array-like
        Original data
    scores : array-like
        Anomaly scores
    labels : array-like, optional
        Ground truth labels (1 for anomaly, 0 for normal)
    predictions : array-like, optional
        Predicted anomalies
    threshold : float, optional
        Anomaly threshold
    save_path : str, optional
        Path to save the figure
    title : str, optional
        Figure title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot data
    ax1.plot(time_idx, data, 'b-', label='Data')
    if labels is not None:
        # Highlight anomaly regions
        ax1.fill_between(time_idx, np.min(data) - 0.1, np.max(data) + 0.1, 
                        where=labels.astype(bool), color='red', alpha=0.2, label='True Anomaly')
    ax1.set_title('Time Series Data')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')
    
    # Plot anomaly scores
    ax2.plot(time_idx, scores, 'g-', label='Anomaly Score')
    if threshold is not None:
        ax2.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
    
    if predictions is not None:
        # Mark predicted anomalies
        anomaly_idx = np.where(predictions.astype(bool))[0]
        if len(anomaly_idx) > 0:
            ax2.scatter(time_idx[anomaly_idx], scores[anomaly_idx], color='red', s=20, label='Predicted Anomaly')
    
    ax2.set_title('Anomaly Scores')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right')
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_attention_weights(attention_weights, labels=None, save_path=None, title=None):
    """
    Plot attention weights as heatmap
    
    Parameters:
    -----------
    attention_weights : array-like
        Attention weights matrix
    labels : list, optional
        Labels for x and y axis
    save_path : str, optional
        Path to save the figure
    title : str, optional
        Figure title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    im = ax.imshow(attention_weights, cmap='viridis')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Weight', rotation=-90, va="bottom")
    
    # Set ticks and labels
    if labels:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
    # Rotate x tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Set title
    if title:
        ax.set_title(title)
    
    # Set axis labels
    ax.set_xlabel('Target')
    ax.set_ylabel('Source')
    
    # Add grid
    ax.set_xticks(np.arange(attention_weights.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(attention_weights.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    
    # Save or show
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_training_history(train_losses, valid_losses, metrics=None, save_path=None, title=None):
    """
    Plot training and validation losses
    
    Parameters:
    -----------
    train_losses : array-like
        Training losses
    valid_losses : array-like
        Validation losses
    metrics : dict, optional
        Dictionary of additional metrics to plot
    save_path : str, optional
        Path to save the figure
    title : str, optional
        Figure title
    """
    # Determine number of subplots
    n_plots = 1 + (1 if metrics else 0)
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    
    # Plot losses
    epochs = np.arange(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[0].plot(epochs, valid_losses, 'r-', label='Validation Loss')
    axes[0].set_title('Loss Curves')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend(loc='upper right')
    
    # Plot additional metrics
    if metrics:
        for i, (metric_name, metric_values) in enumerate(metrics.items()):
            axes[1].plot(epochs, metric_values, label=metric_name)
        
        axes[1].set_title('Metrics')
        axes[1].set_ylabel('Value')
        axes[1].set_xlabel('Epoch')
        axes[1].grid(True, linestyle='--', alpha=0.7)
        axes[1].legend(loc='upper right')
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Save or show
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_model_outputs(model, dataset, device, output_dir='./results', max_samples=5):
    """
    Visualize model outputs on dataset
    
    Parameters:
    -----------
    model : nn.Module
        Trained model
    dataset : torch.utils.data.Dataset
        Dataset to visualize
    device : torch.device
        Computation device
    output_dir : str
        Output directory for saved visualizations
    max_samples : int
        Maximum number of samples to visualize
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    # Get a few samples
    n_samples = min(max_samples, len(dataset))
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        data, time, stable, label = dataset[idx]
        
        # Add batch dimension
        data = data.unsqueeze(0).float().to(device)
        time = time.unsqueeze(0).float().to(device)
        
        # Get model outputs
        with torch.no_grad():
            recon, score, complexity = model(data, time)
            
        # Convert to numpy for plotting
        data_np = data.cpu().numpy()[0]
        recon_np = recon.cpu().numpy()[0]
        score_np = score.cpu().numpy()[0]
        label_np = label.numpy()
        
        # Time indices
        time_idx = np.arange(len(data_np))
        
        # Visualize original vs reconstructed and anomaly scores
        plot_anomaly_detection_results(
            time_idx=time_idx,
            data=data_np[:, 0],  # First feature
            scores=score_np.flatten(),
            labels=label_np.flatten(),
            save_path=os.path.join(output_dir, f'sample_{i+1}_anomaly.png'),
            title=f'Sample {i+1} - Anomaly Detection'
        )
        
        # Visualize MAFD decomposition if available
        try:
            # Extract high and low frequency components
            ph_high, ph_low, _ = model.feature_extractor(data)
            
            # Convert to numpy
            ph_high_np = ph_high.cpu().numpy()
            ph_low_np = ph_low.cpu().numpy()
            
            # Plot decomposition
            plot_decomposition_results(
                time_idx=time_idx,
                original=data_np[:, 0],
                reconstructed=recon_np[:, 0],
                components=[ph_high_np[0], ph_low_np[0]],
                save_path=os.path.join(output_dir, f'sample_{i+1}_decomposition.png'),
                title=f'Sample {i+1} - Signal Decomposition'
            )
        except:
            print(f"Could not visualize decomposition for sample {i+1}")
            
    print(f"Visualizations saved to {output_dir}")