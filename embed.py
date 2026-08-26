import sys
with open('strategy_weights_inline.py', 'r', encoding='utf-8') as f:
    inline_code = f.read()

with open('main.py', 'r', encoding='utf-8') as f:
    main_code = f.read()

# Remove old inline weights if they exist (just in case)
if '_WEIGHTS_B64 =' in main_code:
    print('Already has weights. Need manual replace.')
    sys.exit(1)

insert_marker = '_current_plan = {}        # cached plan for the current day, keyed by player_id'
parts = main_code.split(insert_marker)
new_code = parts[0] + insert_marker + '\n\n# --- BEGIN INLINE WEIGHTS ---\n' + inline_code + '# --- END INLINE WEIGHTS ---\n' + parts[1]

# Modify _try_load_strategy to use it
load_marker = 'return None\n'
new_code = new_code.replace(
    'return None\n',
    'try:\n        return _StrategicLayer(_load_inline_weights())\n    except Exception as e:\n        log(f"[strategy] failed to load inline weights: {e}")\n    return None\n'
)

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(new_code)
print('Successfully embedded weights in main.py')
