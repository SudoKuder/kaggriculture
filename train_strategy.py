"""Top-level training script for the strategic decision layer.

Usage:
    # Quick smoke test (5 episodes)
    python train_strategy.py --episodes 5 --test

    # Short training run (500 episodes, ~20 min)
    python train_strategy.py --episodes 500

    # Full training run (2000 episodes, ~80 min)
    python train_strategy.py --episodes 2000

    # After training, export weights for submission:
    python -m strategy.export_weights strategy_checkpoints/value_net_final.npz -o strategy_weights_inline.py
"""

import argparse
import os
import sys
import time

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy.train import train_value_network
from strategy.value_net import (
    ValueNetTorch, torch_to_numpy, save_weights, load_weights, ValueNetNumpy,
)
from strategy.features import extract_features


def smoke_test():
    """Quick sanity check that all components work end-to-end."""
    print("=" * 60)
    print("SMOKE TEST: verifying all components")
    print("=" * 60)

    # 1. Feature extraction
    print("\n1. Testing feature extraction...")
    from copy import deepcopy
    from kaggle_environments.envs.kaggriculture.kaggriculture import PRODUCTS, CROPS, ANIMALS, MARKET_PARAMS

    tiles = [["LOCKED" if (x >= 5 or y >= 5) else None for x in range(10)] for y in range(10)]
    farm = {
        "money": 3000.0, "farmer": [4, 4], "hands": [],
        "unlocked_quadrants": ["NW"], "hires_today": 0, "tiles": tiles,
    }
    obs = {
        "step": 0, "day": 0, "hour": 0, "player": 0,
        "farms": [farm, deepcopy(farm)],
        "market": {"inventory": {}, "prices": {p: MARKET_PARAMS[p]["base"] for p in PRODUCTS}},
        "town": {"unlocked_shops": []},
        "private": {
            "shed": {p: 0 for p in PRODUCTS + list(ANIMALS)},
            "seeds": {c: 0 for c in CROPS},
            "inventories": [{}],
        },
    }
    features = extract_features(obs)
    from strategy.features import encode_plan
    features = encode_plan(features, None)
    print(f"   Feature vector: shape={features.shape}, "
          f"range=[{features.min():.3f}, {features.max():.3f}]")
    assert features.shape == (78,), f"Expected (78,), got {features.shape}"
    print("   [OK] Feature extraction OK")

    # 2. Candidate generation
    print("\n2. Testing candidate generation...")
    from strategy.candidates import generate_candidates
    candidates = generate_candidates(obs)
    print(f"   Generated {len(candidates)} candidates: "
          f"{[c['label'] for c in candidates]}")
    assert len(candidates) >= 2, "Need at least 2 candidates"
    assert candidates[0]["label"] == "baseline", "First candidate must be baseline"
    print("   [OK] Candidate generation OK")

    # 3. Value network
    print("\n3. Testing value network...")
    import torch
    model = ValueNetTorch()
    x = torch.randn(1, 78)
    y = model(x)
    print(f"   PyTorch forward: input={x.shape} -> output={y.shape}, val={y.item():.4f}")

    # Numpy equivalence
    vnet_np = torch_to_numpy(model)
    y_np = vnet_np.predict(features)
    y_torch = model(torch.tensor(features).unsqueeze(0)).item()
    diff = abs(y_np - y_torch)
    print(f"   Numpy forward: val={y_np:.4f}, PyTorch: val={y_torch:.4f}, diff={diff:.6f}")
    assert diff < 1e-4, f"Numpy/PyTorch mismatch: {diff}"
    print("   [OK] Value network OK")

    # 4. Weight save/load
    print("\n4. Testing weight serialization...")
    os.makedirs("strategy_checkpoints", exist_ok=True)
    test_path = "strategy_checkpoints/test_weights.npz"
    layers = model.export_numpy_weights()
    save_weights(layers, test_path)
    loaded = load_weights(test_path)
    vnet_loaded = ValueNetNumpy(loaded)
    y_loaded = vnet_loaded.predict(features)
    diff2 = abs(y_loaded - y_np)
    print(f"   Save/load roundtrip: diff={diff2:.6f}")
    assert diff2 < 1e-6, f"Save/load mismatch: {diff2}"
    os.remove(test_path)
    print("   [OK] Weight serialization OK")

    # 5. Opponent pool
    print("\n5. Testing opponent pool...")
    from strategy.opponents import OpponentPool
    pool = OpponentPool()
    print(f"   Pool size: {len(pool)} opponents")
    agent, label = pool.sample()
    print(f"   Sampled: {label}")
    assert len(pool) >= 3, f"Expected >= 3 opponents, got {len(pool)}"
    print("   [OK] Opponent pool OK")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Train the strategic decision layer"
    )
    parser.add_argument(
        "--episodes", type=int, default=500,
        help="Number of training episodes (default: 500)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Mini-batch size (default: 32)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--rollout-days", type=int, default=2,
        help="Rollout depth in days (default: 2)"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=100,
        help="Save checkpoint every N episodes (default: 100)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="strategy_checkpoints",
        help="Checkpoint directory (default: strategy_checkpoints)"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run smoke test instead of training"
    )
    args = parser.parse_args()

    if args.test:
        smoke_test()
        return

    print("=" * 60)
    print("STRATEGIC LAYER TRAINING")
    print("=" * 60)
    print(f"Episodes:   {args.episodes}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR:         {args.lr}")
    print(f"Rollout:    {args.rollout_days} days")
    print(f"Output:     {args.output_dir}")
    print("=" * 60)

    t0 = time.time()
    model = train_value_network(
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        lr=args.lr,
        rollout_depth_days=args.rollout_days,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=args.output_dir,
        verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\nTotal training time: {elapsed / 60:.1f} minutes")
    print(f"Weights saved to: {args.output_dir}/value_net_final.npz")
    print("\nNext steps:")
    print(f"  1. Run benchmark: python benchmark.py")
    print(f"  2. Export for submission:")
    print(f"     python -m strategy.export_weights {args.output_dir}/value_net_final.npz -o strategy_weights_inline.py")


if __name__ == "__main__":
    main()
