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
        print(f"Log file not found at {log_file}")
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

def plot_metrics(metrics, eval_metrics=None, output_file=None):
    """Generate and display/save plots for training metrics."""
    if not metrics:
        print("No metrics to plot.")
        return

    episodes = [m['episode'] for m in metrics]
        
    losses = [m['loss'] for m in metrics]
    actor_losses = [m.get('actor_loss') for m in metrics if 'actor_loss' in m]
    critic_losses = [m.get('critic_loss') for m in metrics if 'critic_loss' in m]
    entropy_losses = [m.get('entropy_loss') for m in metrics if 'entropy_loss' in m]
    
    deltas = [m['money_delta'] for m in metrics]
    buffer_sizes = [m.get('buffer_size', 0) for m in metrics]
    eps_per_secs = [m.get('eps_per_sec', 0) for m in metrics]
    opponents = [m.get('opponent', 'unknown') for m in metrics]
    
    # Calculate moving average for deltas to smooth out opponent noise
    window = min(50, max(1, len(deltas) // 10))
    smoothed_deltas = moving_average(deltas, window)
    smoothed_episodes = episodes[window-1:]

    # Create figure with 6 subplots
    fig, axs = plt.subplots(6, 1, figsize=(12, 22), sharex=True)
    fig.suptitle('Strategic Layer Training Metrics', fontsize=18, y=0.98)

    # 1. Plot Loss
    if len(losses) > window:
        smoothed_losses = moving_average(losses, window)
        axs[0].plot(episodes, losses, color='grey', linewidth=1.0, alpha=0.3, label='Total Loss')
        axs[0].plot(smoothed_episodes, smoothed_losses, color='black', linewidth=2.0, label=f'Total Avg (n={window})')
    else:
        axs[0].plot(episodes, losses, color='black', linewidth=1.5, alpha=0.8, label='Total Loss')
        
    if len(actor_losses) == len(losses):
        axs[0].plot(episodes, actor_losses, 'tab:blue', alpha=0.5, label='Actor Loss')
    if len(critic_losses) == len(losses):
        axs[0].plot(episodes, critic_losses, 'tab:red', alpha=0.5, label='Critic Loss (MSE)')
    if len(entropy_losses) == len(losses):
        axs[0].plot(episodes, entropy_losses, 'tab:green', alpha=0.5, label='Entropy Loss')
        
    axs[0].legend(loc='upper right', fontsize='small')
    axs[0].set_ylabel('Loss Value')
    axs[0].set_title('Network Losses')
    axs[0].grid(True, alpha=0.3)
    if max(losses) > 100 * max(0.1, min([l for l in losses if l > 0] + [1])):
        axs[0].set_yscale('symlog')

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
    axs[1].set_title('Training Performance vs Opponents (Stochastic)')
    axs[1].legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize='small', title="Opponents")
    axs[1].grid(True, alpha=0.3)

    # 3. & 4. Evaluation Metrics
    if eval_metrics:
        eval_eps = sorted(list(set([m['episode'] for m in eval_metrics])))
        eval_opps = sorted(list(set([m['opponent'] for m in eval_metrics])))
        e_colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(eval_opps))))
        
        for i, opp in enumerate(eval_opps):
            o_eps = [m['episode'] for m in eval_metrics if m['opponent'] == opp]
            o_wins = [m['win_rate'] for m in eval_metrics if m['opponent'] == opp]
            o_md = [m['mean_money_delta'] for m in eval_metrics if m['opponent'] == opp]
            
            axs[2].plot(o_eps, o_wins, marker='o', linestyle='-', color=e_colors[i % len(e_colors)], label=opp)
            axs[3].plot(o_eps, o_md, marker='s', linestyle='-', color=e_colors[i % len(e_colors)], label=opp)
            
    axs[2].axhline(0.5, color='red', linestyle='--', alpha=0.5)
    axs[2].set_ylabel('Win Rate')
    axs[2].set_title('Evaluation Win Rate (Deterministic)')
    axs[2].grid(True, alpha=0.3)
    axs[2].set_ylim([-0.05, 1.05])
    if eval_metrics:
        axs[2].legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize='small', title="Eval Opponents")
        
    axs[3].axhline(0, color='red', linestyle='--', alpha=0.5)
    axs[3].set_ylabel('Mean Money Delta')
    axs[3].set_title('Evaluation Mean Money Delta (Deterministic)')
    axs[3].grid(True, alpha=0.3)

    # 5. Plot Buffer Size
    axs[4].plot(episodes, buffer_sizes, 'tab:purple', linewidth=2)
    axs[4].set_ylabel('Buffer Size')
    axs[4].set_title('Replay Buffer Size')
    axs[4].grid(True, alpha=0.3)

    # 6. Plot Episodes/Sec
    if len(eps_per_secs) > window:
        smoothed_eps = moving_average(eps_per_secs, window)
        axs[5].plot(episodes, eps_per_secs, 'tab:green', linewidth=1.0, alpha=0.3)
        axs[5].plot(smoothed_episodes, smoothed_eps, 'tab:green', linewidth=2.5, label=f'Moving Avg (n={window})')
        axs[5].legend(loc='upper right', fontsize='small')
    else:
        axs[5].plot(episodes, eps_per_secs, 'tab:green', linewidth=1.5, alpha=0.8)
        
    axs[5].set_ylabel('Episodes / Sec')
    axs[5].set_xlabel('Episode')
    axs[5].set_title('Training Speed')
    axs[5].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 0.9, 0.97])  # Leave room for the legend on the right
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize training logs")
    parser.add_argument("log", nargs="?", default="training_metrics.jsonl", 
                        help="Path to the JSONL log file")
    parser.add_argument("--output", "-o", default="training_graphs.png", 
                        help="Output image file (e.g., graph.png)")
    parser.add_argument("--show", action="store_true", 
                        help="Display the plot interactively instead of just saving")
    
    args = parser.parse_args()
    
    log_path = args.log
    if not log_path.endswith('.jsonl') and not os.path.exists(log_path):
        if os.path.exists(log_path + '.jsonl'):
            log_path += '.jsonl'
            
    eval_log_path = os.path.join(os.path.dirname(log_path), "eval_metrics.jsonl")
    
    metrics = load_metrics(log_path)
    eval_metrics = load_metrics(eval_log_path)
    
    print(f"Loaded {len(metrics)} training metric records from {log_path}")
    if eval_metrics:
        print(f"Loaded {len(eval_metrics)} evaluation metric records from {eval_log_path}")
    else:
        print("No evaluation metrics found.")
    
    # Determine output file based on arguments
    out_file = None if args.show and not args.output else args.output
    
    plot_metrics(metrics, eval_metrics, out_file)
    
    if args.show and args.output:
        plt.show()
