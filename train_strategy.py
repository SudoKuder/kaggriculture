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

from strategy.train import train_actor_network
from strategy.actor_net import (
    ActorNetTorch, torch_to_numpy, save_weights, load_weights, ActorNetNumpy,
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
    features = extract_features(obs)
    print(f"   Feature vector: shape={features.shape}, "
          f"range=[{features.min():.3f}, {features.max():.3f}]")
    assert features.shape == (64,), f"Expected (64,), got {features.shape}"
    print("   [OK] Feature extraction OK")

    # 2. Actor network (forward pass + action sampling)
    print("\n2. Testing actor network...")
    import torch
    model = ActorNetTorch()
    x = torch.randn(1, 64)
    y = model(x)
    print(f"   PyTorch forward: input={x.shape} -> output={y.shape}")

    # Test get_action returns all expected keys
    action_dict, log_prob, logits = model.get_action(
        torch.tensor(features).unsqueeze(0), deterministic=True
    )
    expected_keys = {
        "buy_land", "hire_target", "sell_hold", "buy_seed_crop",
        "buy_seed_frac", "buy_animal_type", "buy_animal_frac",
        "weed_penalty", "maint_water_hour", "panic_drop_hour",
        "seed_threshold_mult", "land_unlock_buffer",
    }
    assert set(action_dict.keys()) == expected_keys, (
        f"Missing keys: {expected_keys - set(action_dict.keys())}"
    )
    print(f"   get_action: {len(action_dict)} heads, log_prob shape={log_prob.shape}")
    print("   [OK] Actor network OK")

    # 3. Numpy equivalence
    print("\n3. Testing numpy inference equivalence...")
    vnet_np = torch_to_numpy(model)
    action_np = vnet_np.predict(features)
    diff = abs(
        action_np["buy_seed_frac"]
        - torch.sigmoid(model(torch.tensor(features).unsqueeze(0))[..., 28]).item()
    )
    print(f"   Numpy forward: val={action_np['buy_seed_frac']:.4f}, PyTorch diff={diff:.6f}")
    assert diff < 1e-4, f"Numpy/PyTorch mismatch: {diff}"
    print("   [OK] Numpy equivalence OK")

    # 4. Weight save/load
    print("\n4. Testing weight serialization...")
    os.makedirs("strategy_checkpoints", exist_ok=True)
    test_path = "strategy_checkpoints/test_weights.npz"
    layers = model.export_numpy_weights()
    save_weights(layers, test_path)
    loaded = load_weights(test_path)
    vnet_loaded = ActorNetNumpy(loaded)
    action_loaded = vnet_loaded.predict(features)
    diff2 = abs(action_loaded["buy_seed_frac"] - action_np["buy_seed_frac"])
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
    print(f"Output:     {args.output_dir}")
    print("=" * 60)

    t0 = time.time()
    model = train_actor_network(
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=args.output_dir,
        verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\nTotal training time: {elapsed / 60:.1f} minutes")
    print(f"Weights saved to: {args.output_dir}/actor_net_final.npz")
    print("\nNext steps:")
    print(f"  1. Run benchmark: python benchmark.py")
    print(f"  2. Export for submission:")
    print(f"     python -m strategy.export_weights {args.output_dir}/actor_net_final.npz -o strategy_weights_inline.py")


if __name__ == "__main__":
    main()
