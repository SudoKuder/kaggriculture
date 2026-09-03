import re

with open('strategy/train.py', 'r') as f:
    content = f.read()

# 1. Imports
content = content.replace(
    "from strategy.actor_net import ActorNetTorch, ActorNetNumpy, torch_to_numpy, save_weights",
    "from strategy.actor_net import ActorNetTorch, ActorNetNumpy, torch_to_numpy, save_weights\nfrom strategy.value_net import ValueNetTorch"
)

# 2. Model Initialization
content = content.replace(
    "model = ActorNetTorch(input_dim=FEATURE_DIM).to(device)",
    "model = ActorNetTorch(input_dim=FEATURE_DIM).to(device)\n    critic = ValueNetTorch(input_dim=FEATURE_DIM).to(device)"
)

# 3. Checkpoints and Optimizer
old_ckpt_logic = """
    if os.path.exists(final_path):
        loaded_path = final_path
    elif checkpoints:
        loaded_path = max(checkpoints, key=extract_ep)
        
    if loaded_path:
        if verbose:
            print(f"Resuming training from weights at {loaded_path} (Starting at Episode {start_episode})")
        from strategy.actor_net import load_weights
        layers = load_weights(loaded_path)
        model.import_numpy_weights(layers)

    optimizer = optim.Adam(model.parameters(), lr=lr)
"""
new_ckpt_logic = """
    loaded_path = None
    loaded_critic_path = None
    if os.path.exists(final_path):
        loaded_path = final_path
        critic_final_path = os.path.join(output_dir, "critic_net_final.npz")
        if os.path.exists(critic_final_path):
            loaded_critic_path = critic_final_path
    elif checkpoints:
        loaded_path = max(checkpoints, key=extract_ep)
        loaded_critic_path = os.path.join(output_dir, f"critic_net_ep{start_episode}.npz")
        
    if loaded_path:
        if verbose:
            print(f"Resuming training from weights at {loaded_path} (Starting at Episode {start_episode})")
        from strategy.actor_net import load_weights
        layers = load_weights(loaded_path)
        model.import_numpy_weights(layers)
        
        if loaded_critic_path and os.path.exists(loaded_critic_path):
            from strategy.value_net import load_weights as load_critic_weights
            c_layers = load_critic_weights(loaded_critic_path)
            critic.import_numpy_weights(c_layers)

    optimizer = optim.Adam(list(model.parameters()) + list(critic.parameters()), lr=lr)
"""
content = content.replace(old_ckpt_logic.strip(), new_ckpt_logic.strip())

# 4. Loss computation
old_loss_logic = """
                # Rewards (baseline could be subtracted here, e.g., running mean of money_delta)
                # But simple REINFORCE with log-money works as a start
                batch_r = torch.tensor(
                    np.array([_log_money(data_targets[i]) for i in indices]),
                    dtype=torch.float32,
                ).to(device)
                
                # Loss is -log_prob * reward
                loss = -(total_log_prob * batch_r).mean()
    
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
"""
new_loss_logic = """
                batch_r = torch.tensor(
                    np.array([_log_money(data_targets[i]) for i in indices]),
                    dtype=torch.float32,
                ).to(device)
                
                batch_v = critic(batch_x)
                advantages = batch_r - batch_v.detach()
                
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                actor_loss = -(total_log_prob * advantages).mean()
                critic_loss = nn.functional.mse_loss(batch_v, batch_r)
                
                loss = actor_loss + 0.5 * critic_loss
    
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                optimizer.step()
"""
content = content.replace(old_loss_logic.strip(), new_loss_logic.strip())

# 5. Checkpoint saving logic
old_save_ckpt = """
        if (episode + 1) % checkpoint_interval == 0:
            ckpt_path = os.path.join(output_dir, f"actor_net_ep{episode + 1}.npz")
            layers = model.export_numpy_weights()
            save_weights(layers, ckpt_path)
            if verbose:
                print(f"  → Checkpoint saved: {ckpt_path}")
"""
new_save_ckpt = """
        if (episode + 1) % checkpoint_interval == 0:
            ckpt_path = os.path.join(output_dir, f"actor_net_ep{episode + 1}.npz")
            critic_ckpt_path = os.path.join(output_dir, f"critic_net_ep{episode + 1}.npz")
            
            layers = model.export_numpy_weights()
            save_weights(layers, ckpt_path)
            
            from strategy.value_net import save_weights as save_critic_weights
            c_layers = critic.export_numpy_weights()
            save_critic_weights(c_layers, critic_ckpt_path)
            
            if verbose:
                print(f"  → Checkpoint saved: {ckpt_path} and {critic_ckpt_path}")
"""
content = content.replace(old_save_ckpt.strip(), new_save_ckpt.strip())

# 6. Final saving logic
old_save_final = """
    final_path = os.path.join(output_dir, "actor_net_final.npz")
    layers = model.export_numpy_weights()
    save_weights(layers, final_path)
"""
new_save_final = """
    final_path = os.path.join(output_dir, "actor_net_final.npz")
    critic_final_path = os.path.join(output_dir, "critic_net_final.npz")
    layers = model.export_numpy_weights()
    save_weights(layers, final_path)
    
    from strategy.value_net import save_weights as save_critic_weights
    c_layers = critic.export_numpy_weights()
    save_critic_weights(c_layers, critic_final_path)
"""
content = content.replace(old_save_final.strip(), new_save_final.strip())


with open('strategy/train.py', 'w') as f:
    f.write(content)

print("Updated train.py successfully!")
