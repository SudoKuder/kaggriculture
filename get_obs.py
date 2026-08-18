import json

with open('view_game.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

html = ""
for c in nb.get('cells', []):
    if c.get('outputs'):
        for o in c['outputs']:
            if 'data' in o and 'text/html' in o['data']:
                html += "".join(o['data']['text/html'])

idx = html.find('"observation": {')
if idx != -1:
    idx += 15
    stack = []
    end = -1
    for i, c in enumerate(html[idx:]):
        if c == '{':
            stack.append(c)
        elif c == '}':
            stack.pop()
            if not stack:
                end = i
                break
    
    if end != -1:
        obs_str = html[idx:idx+end+1]
        with open('obs.json', 'w') as f:
            f.write(obs_str)
