"""Visualize training metrics from the strategic decision layer.

Reads the JSONL logs generated during training and plots:
1. Value Network Loss over time
2. Money Delta (Agent - Opponent) over time
3. Moving average of Money Delta
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(log_file):
    """Load metrics from JSONL file."""
    if not os.path.exists(log_file):
        print(f"Error: Log file not found at {log_file}")
        return []
        
    metrics = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                metrics.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return metrics

def moving_average(a, n=10):
    """Calculate moving average."""
    if len(a) < n:
        return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_metrics(metrics, output_file=None):
    """Generate and display/save plots for training metrics."""
    if not metrics:
        print("No metrics to plot.")
        return

    episodes = [m['episode'] for m in metrics]
    losses = [m['loss'] for m in metrics]
    deltas = [m['money_delta'] for m in metrics]
    
    # Calculate moving average for deltas to smooth out opponent noise
    window = min(10, max(1, len(deltas) // 5))
    smoothed_deltas = moving_average(deltas, window)
    smoothed_episodes = episodes[window-1:]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Strategic Layer Training Metrics', fontsize=16)

    # Plot Loss
    ax1.plot(episodes, losses, 'tab:red', linewidth=2)
    ax1.set_ylabel('MSE Loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_title('Value Network Loss')
    ax1.grid(True, alpha=0.3)
    
    # Use log scale for loss if values vary significantly
    if max(losses) > 100 * max(0.1, min([l for l in losses if l > 0] + [1])):
        ax1.set_yscale('log')

    # Plot Money Delta
    ax2.scatter(episodes, deltas, alpha=0.4, color='tab:blue', label='Raw Delta')
    if len(smoothed_deltas) > 0:
        ax2.plot(smoothed_episodes, smoothed_deltas, 'tab:blue', linewidth=3, label=f'Moving Avg (n={window})')
    
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Money Delta vs Opponent', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_title('Agent Performance (Higher is better)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize training logs")
    parser.add_argument("--log", default="strategy_checkpoints/logs/training_metrics.jsonl", 
                        help="Path to the JSONL log file")
    parser.add_argument("--output", "-o", default="training_graphs.png", 
                        help="Output image file (e.g., graph.png)")
    parser.add_argument("--show", action="store_true", 
                        help="Display the plot interactively instead of just saving")
    
    args = parser.parse_args()
    
    metrics = load_metrics(args.log)
    print(f"Loaded {len(metrics)} metric records from {args.log}")
    
    # Determine output file based on arguments
    out_file = None if args.show and not args.output else args.output
    
    plot_metrics(metrics, out_file)
    
    if args.show and args.output:
        plt.show()
