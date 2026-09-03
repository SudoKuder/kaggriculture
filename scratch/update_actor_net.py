import re

with open('strategy/actor_net.py', 'r') as f:
    content = f.read()

# 1. Update ACTION_DIM and add comments
content = content.replace("ACTION_DIM = 26", "ACTION_DIM = 31")
old_comments = """
# [25] buy_animal_frac logit (sigmoid)
"""
new_comments = """
# [25] buy_animal_frac logit (sigmoid)
# [26] weed_penalty logit (sigmoid -> scale [0, 10])
# [27] maint_water_hour logit (sigmoid -> scale [0, 23])
# [28] panic_drop_hour logit (sigmoid -> scale [0, 23])
# [29] seed_threshold_mult logit (sigmoid -> scale [0, 5])
# [30] land_unlock_buffer logit (sigmoid -> scale [0, 2000])
"""
content = content.replace(old_comments.strip(), new_comments.strip())

# 2. Update get_action
old_get_action = """
            # 7. buy_animal_frac (Normal)
            mean_anim_f = torch.sigmoid(logits[..., 25])
            dist_anim_f = Normal(mean_anim_f, std)
            if deterministic:
                a_anim_f = mean_anim_f
            else:
                a_anim_f = dist_anim_f.sample().clamp(0.0, 1.0)
            lp_anim_f = dist_anim_f.log_prob(a_anim_f)
            
            total_log_prob = lp_land + lp_hire + lp_sell + lp_seed_c + lp_seed_f + lp_anim_t + lp_anim_f
            
            action_dict = {
                "buy_land": a_land,
                "hire_target": a_hire,
                "sell_hold": a_sell,
                "buy_seed_crop": a_seed_c,
                "buy_seed_frac": a_seed_f,
                "buy_animal_type": a_anim_t,
                "buy_animal_frac": a_anim_f,
            }
"""

new_get_action = """
            # 7. buy_animal_frac (Normal)
            mean_anim_f = torch.sigmoid(logits[..., 25])
            dist_anim_f = Normal(mean_anim_f, std)
            if deterministic:
                a_anim_f = mean_anim_f
            else:
                a_anim_f = dist_anim_f.sample().clamp(0.0, 1.0)
            lp_anim_f = dist_anim_f.log_prob(a_anim_f)
            
            # 8. Tactical parameters (Normal distributions)
            mean_weed = torch.sigmoid(logits[..., 26]) * 10.0
            dist_weed = Normal(mean_weed, 1.0)
            a_weed = mean_weed if deterministic else dist_weed.sample().clamp(0.0, 10.0)
            lp_weed = dist_weed.log_prob(a_weed)
            
            mean_maint = torch.sigmoid(logits[..., 27]) * 23.0
            dist_maint = Normal(mean_maint, 2.0)
            a_maint = mean_maint if deterministic else dist_maint.sample().clamp(0.0, 23.0)
            lp_maint = dist_maint.log_prob(a_maint)
            
            mean_panic = torch.sigmoid(logits[..., 28]) * 23.0
            dist_panic = Normal(mean_panic, 2.0)
            a_panic = mean_panic if deterministic else dist_panic.sample().clamp(0.0, 23.0)
            lp_panic = dist_panic.log_prob(a_panic)
            
            mean_seed_m = torch.sigmoid(logits[..., 29]) * 5.0
            dist_seed_m = Normal(mean_seed_m, 0.5)
            a_seed_m = mean_seed_m if deterministic else dist_seed_m.sample().clamp(0.0, 5.0)
            lp_seed_m = dist_seed_m.log_prob(a_seed_m)
            
            mean_land_b = torch.sigmoid(logits[..., 30]) * 2000.0
            dist_land_b = Normal(mean_land_b, 200.0)
            a_land_b = mean_land_b if deterministic else dist_land_b.sample().clamp(0.0, 2000.0)
            lp_land_b = dist_land_b.log_prob(a_land_b)
            
            total_log_prob = lp_land + lp_hire + lp_sell + lp_seed_c + lp_seed_f + lp_anim_t + lp_anim_f + lp_weed + lp_maint + lp_panic + lp_seed_m + lp_land_b
            
            action_dict = {
                "buy_land": a_land,
                "hire_target": a_hire,
                "sell_hold": a_sell,
                "buy_seed_crop": a_seed_c,
                "buy_seed_frac": a_seed_f,
                "buy_animal_type": a_anim_t,
                "buy_animal_frac": a_anim_f,
                "weed_penalty": a_weed,
                "maint_water_hour": a_maint,
                "panic_drop_hour": a_panic,
                "seed_threshold_mult": a_seed_m,
                "land_unlock_buffer": a_land_b,
            }
"""
content = content.replace(old_get_action.strip(), new_get_action.strip())

# 3. Update ActorNetNumpy
old_numpy = """
        return {
            "buy_land": self._sigmoid(logits[0]),
            "hire_target": np.argmax(logits[1:7]),
            "sell_hold": self._sigmoid(logits[7:16]),
            "buy_seed_crop": np.argmax(logits[16:21]),
            "buy_seed_frac": self._sigmoid(logits[21]),
            "buy_animal_type": np.argmax(logits[22:25]),
            "buy_animal_frac": self._sigmoid(logits[25]),
        }
"""
new_numpy = """
        return {
            "buy_land": self._sigmoid(logits[0]),
            "hire_target": np.argmax(logits[1:7]),
            "sell_hold": self._sigmoid(logits[7:16]),
            "buy_seed_crop": np.argmax(logits[16:21]),
            "buy_seed_frac": self._sigmoid(logits[21]),
            "buy_animal_type": np.argmax(logits[22:25]),
            "buy_animal_frac": self._sigmoid(logits[25]),
            "weed_penalty": self._sigmoid(logits[26]) * 10.0,
            "maint_water_hour": self._sigmoid(logits[27]) * 23.0,
            "panic_drop_hour": self._sigmoid(logits[28]) * 23.0,
            "seed_threshold_mult": self._sigmoid(logits[29]) * 5.0,
            "land_unlock_buffer": self._sigmoid(logits[30]) * 2000.0,
        }
"""
content = content.replace(old_numpy.strip(), new_numpy.strip())

with open('strategy/actor_net.py', 'w') as f:
    f.write(content)
