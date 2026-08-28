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
    epsilons = [m.get('epsilon', 0) for m in metrics]
    buffer_sizes = [m.get('buffer_size', 0) for m in metrics]
    eps_per_secs = [m.get('eps_per_sec', 0) for m in metrics]
    opponents = [m.get('opponent', 'unknown') for m in metrics]
    
    # Calculate moving average for deltas to smooth out opponent noise
    window = min(50, max(1, len(deltas) // 10))
    smoothed_deltas = moving_average(deltas, window)
    smoothed_episodes = episodes[window-1:]

    # Create figure with 5 subplots
    fig, axs = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
    fig.suptitle('Strategic Layer Training Metrics', fontsize=18, y=0.98)

    # 1. Plot Loss
    if len(losses) > window:
        smoothed_losses = moving_average(losses, window)
        axs[0].plot(episodes, losses, 'tab:red', linewidth=1.0, alpha=0.3)
        axs[0].plot(smoothed_episodes, smoothed_losses, 'tab:red', linewidth=2.5, label=f'Moving Avg (n={window})')
        axs[0].legend(loc='upper right', fontsize='small')
    else:
        axs[0].plot(episodes, losses, 'tab:red', linewidth=1.5, alpha=0.8)
        
    axs[0].set_ylabel('MSE Loss')
    axs[0].set_title('Value Network Loss')
    axs[0].grid(True, alpha=0.3)
    if max(losses) > 100 * max(0.1, min([l for l in losses if l > 0] + [1])):
        axs[0].set_yscale('log')

    # 2. Plot Money Delta Colored by Opponent
    unique_opps = sorted(list(set(opponents)))
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(unique_opps))))
    for i, opp in enumerate(unique_opps):
        opp_episodes = [e for e, o in zip(episodes, opponents) if o == opp]
        opp_deltas = [d for d, o in zip(deltas, opponents) if o == opp]
        axs[1].scatter(opp_episodes, opp_deltas, alpha=0.5, color=colors[i % len(colors)], label=opp, s=15)
        
    if len(smoothed_deltas) > 0:
        axs[1].plot(smoothed_episodes, smoothed_deltas, 'black', linewidth=3, label=f'Moving Avg (n={window})')
    
    axs[1].axhline(0, color='red', linestyle='--', alpha=0.5)
    axs[1].set_ylabel('Money Delta')
    axs[1].set_title('Agent Performance vs Opponents (Higher is better)')
    axs[1].legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize='small', title="Opponents")
    axs[1].grid(True, alpha=0.3)

    # 3. Plot Epsilon
    axs[2].plot(episodes, epsilons, 'tab:orange', linewidth=2)
    axs[2].set_ylabel('Epsilon')
    axs[2].set_title('Exploration Rate (Epsilon-Greedy)')
    axs[2].grid(True, alpha=0.3)
    
    # 4. Plot Buffer Size
    axs[3].plot(episodes, buffer_sizes, 'tab:purple', linewidth=2)
    axs[3].set_ylabel('Buffer Size')
    axs[3].set_title('Replay Buffer Size')
    axs[3].grid(True, alpha=0.3)

    # 5. Plot Episodes/Sec
    if len(eps_per_secs) > window:
        smoothed_eps = moving_average(eps_per_secs, window)
        axs[4].plot(episodes, eps_per_secs, 'tab:green', linewidth=1.0, alpha=0.3)
        axs[4].plot(smoothed_episodes, smoothed_eps, 'tab:green', linewidth=2.5, label=f'Moving Avg (n={window})')
        axs[4].legend(loc='upper right', fontsize='small')
    else:
        axs[4].plot(episodes, eps_per_secs, 'tab:green', linewidth=1.5, alpha=0.8)
        
    axs[4].set_ylabel('Episodes / Sec')
    axs[4].set_xlabel('Episode')
    axs[4].set_title('Training Speed')
    axs[4].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 0.9, 0.97])  # Leave room for the legend on the right
    
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
