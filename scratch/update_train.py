import re

with open('strategy/train.py', 'r') as f:
    content = f.read()

# 1. Update epsilon-greedy random action
old_epsilon = """
                    a_seed_c = random.randint(0, 4)
                    a_seed_f = random.random()
                    a_anim_t = random.randint(0, 2)
                    a_anim_f = random.random()
                    
                    action_dict = {
                        "buy_land": torch.tensor([a_land]),
                        "hire_target": torch.tensor([a_hire]),
                        "sell_hold": torch.tensor([a_sell]),
                        "buy_seed_crop": torch.tensor([a_seed_c]),
                        "buy_seed_frac": torch.tensor([a_seed_f]),
                        "buy_animal_type": torch.tensor([a_anim_t]),
                        "buy_animal_frac": torch.tensor([a_anim_f]),
                    }
"""
new_epsilon = """
                    a_seed_c = random.randint(0, 4)
                    a_seed_f = random.random()
                    a_anim_t = random.randint(0, 2)
                    a_anim_f = random.random()
                    a_weed = random.random() * 10.0
                    a_maint = random.random() * 23.0
                    a_panic = random.random() * 23.0
                    a_seed_m = random.random() * 5.0
                    a_land_b = random.random() * 2000.0
                    
                    action_dict = {
                        "buy_land": torch.tensor([a_land]),
                        "hire_target": torch.tensor([a_hire]),
                        "sell_hold": torch.tensor([a_sell]),
                        "buy_seed_crop": torch.tensor([a_seed_c]),
                        "buy_seed_frac": torch.tensor([a_seed_f]),
                        "buy_animal_type": torch.tensor([a_anim_t]),
                        "buy_animal_frac": torch.tensor([a_anim_f]),
                        "weed_penalty": torch.tensor([a_weed]),
                        "maint_water_hour": torch.tensor([a_maint]),
                        "panic_drop_hour": torch.tensor([a_panic]),
                        "seed_threshold_mult": torch.tensor([a_seed_m]),
                        "land_unlock_buffer": torch.tensor([a_land_b]),
                    }
"""
content = content.replace(old_epsilon.strip(), new_epsilon.strip())

# 2. Update backprop collect actions
old_collect = """
                a_seed_c = torch.tensor(np.array([data_actions[i]["buy_seed_crop"] for i in indices])).to(device)
                a_seed_f = torch.tensor(np.array([data_actions[i]["buy_seed_frac"] for i in indices])).to(device)
                a_anim_t = torch.tensor(np.array([data_actions[i]["buy_animal_type"] for i in indices])).to(device)
                a_anim_f = torch.tensor(np.array([data_actions[i]["buy_animal_frac"] for i in indices])).to(device)
"""
new_collect = """
                a_seed_c = torch.tensor(np.array([data_actions[i]["buy_seed_crop"] for i in indices])).to(device)
                a_seed_f = torch.tensor(np.array([data_actions[i]["buy_seed_frac"] for i in indices])).to(device)
                a_anim_t = torch.tensor(np.array([data_actions[i]["buy_animal_type"] for i in indices])).to(device)
                a_anim_f = torch.tensor(np.array([data_actions[i]["buy_animal_frac"] for i in indices])).to(device)
                a_weed = torch.tensor(np.array([data_actions[i]["weed_penalty"] for i in indices])).to(device)
                a_maint = torch.tensor(np.array([data_actions[i]["maint_water_hour"] for i in indices])).to(device)
                a_panic = torch.tensor(np.array([data_actions[i]["panic_drop_hour"] for i in indices])).to(device)
                a_seed_m = torch.tensor(np.array([data_actions[i]["seed_threshold_mult"] for i in indices])).to(device)
                a_land_b = torch.tensor(np.array([data_actions[i]["land_unlock_buffer"] for i in indices])).to(device)
"""
content = content.replace(old_collect.strip(), new_collect.strip())

# 3. Update log_prob computation
old_log_prob = """
                lp_seed_c = Categorical(logits=logits[..., 16:21]).log_prob(a_seed_c)
                lp_seed_f = Normal(torch.sigmoid(logits[..., 21]), 0.2).log_prob(a_seed_f)
                lp_anim_t = Categorical(logits=logits[..., 22:25]).log_prob(a_anim_t)
                lp_anim_f = Normal(torch.sigmoid(logits[..., 25]), 0.2).log_prob(a_anim_f)
                
                total_log_prob = lp_land + lp_hire + lp_sell + lp_seed_c + lp_seed_f + lp_anim_t + lp_anim_f
"""
new_log_prob = """
                lp_seed_c = Categorical(logits=logits[..., 16:21]).log_prob(a_seed_c)
                lp_seed_f = Normal(torch.sigmoid(logits[..., 21]), 0.2).log_prob(a_seed_f)
                lp_anim_t = Categorical(logits=logits[..., 22:25]).log_prob(a_anim_t)
                lp_anim_f = Normal(torch.sigmoid(logits[..., 25]), 0.2).log_prob(a_anim_f)
                lp_weed = Normal(torch.sigmoid(logits[..., 26]) * 10.0, 1.0).log_prob(a_weed)
                lp_maint = Normal(torch.sigmoid(logits[..., 27]) * 23.0, 2.0).log_prob(a_maint)
                lp_panic = Normal(torch.sigmoid(logits[..., 28]) * 23.0, 2.0).log_prob(a_panic)
                lp_seed_m = Normal(torch.sigmoid(logits[..., 29]) * 5.0, 0.5).log_prob(a_seed_m)
                lp_land_b = Normal(torch.sigmoid(logits[..., 30]) * 2000.0, 200.0).log_prob(a_land_b)
                
                total_log_prob = lp_land + lp_hire + lp_sell + lp_seed_c + lp_seed_f + lp_anim_t + lp_anim_f + lp_weed + lp_maint + lp_panic + lp_seed_m + lp_land_b
"""
content = content.replace(old_log_prob.strip(), new_log_prob.strip())

with open('strategy/train.py', 'w') as f:
    f.write(content)
