import re

with open('strategy_weights_inline.py', 'r') as f:
    inline_content = f.read()
match = re.search(r'_WEIGHTS_B64 = "(.*?)"', inline_content)
new_b64 = match.group(1)

with open('main.py', 'r') as f:
    main_content = f.read()

new_main = re.sub(r'_WEIGHTS_B64 = "(.*?)"', f'_WEIGHTS_B64 = "{new_b64}"', main_content)

with open('main.py', 'w') as f:
    f.write(new_main)
print('Updated main.py inline weights!')
