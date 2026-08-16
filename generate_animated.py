"""
Creates an animated game board viewer by replaying the full game
"""
import json
from kaggle_environments import make

# Load action log
with open("game_log.json", "r", encoding="utf-8") as f:
    data = json.load(f)

actions = data['action_log']

# Replay the game to capture board state at each step
env = make("kaggriculture")
obs = env.reset()

# Capture states
states = []

def capture_state():
    """Capture current board state"""
    o0 = obs[0]['observation']
    o1 = obs[1]['observation']
    
    boards = o0['farms'][0]['tiles'], o1['farms'][1]['tiles']
    money = o0['farms'][0]['money'], o1['farms'][1]['money']
    farmer_pos = o0['farms'][0]['farmer'], o1['farms'][1]['farmer']
    
    return {
        'step': len(states),
        'boards': [b.tolist() if hasattr(b, 'tolist') else b for b in boards],
        'money': list(money),
        'farmer_pos': [list(pos) for pos in farmer_pos],
    }

# Group actions by step
actions_by_step = {}
for action in actions:
    step = action['step']
    if step not in actions_by_step:
        actions_by_step[step] = [None, None]
    actions_by_step[step][action['player']] = action

# Initial capture
states.append(capture_state())

# Replay each step
for step_num in range(1, 720):
    step_actions = actions_by_step.get(step_num, [None, None])
    
    # Build action dict from recorded actions
    action_dict = {"farmer": ["PASS"], "hands": [], "market": []}
    
    # Step environment with the recorded action
    try:
        result = env.step([action_dict, action_dict])
        # Build action dict for each player
        actions_p0 = {"farmer": ["PASS"], "hands": [], "market": []}
        actions_p1 = {"farmer": ["PASS"], "hands": [], "market": []}
        
        if step_actions[0]:
            actions_p0["farmer"] = step_actions[0]['farmer_action']
            actions_p0["market"] = step_actions[0]['market_orders']
        if step_actions[1]:
            actions_p1["farmer"] = step_actions[1]['farmer_action']
            actions_p1["market"] = step_actions[1]['market_orders']
        
        result = env.step([actions_p0, actions_p1])
        obs = result
        states.append(capture_state())
    except Exception as e:
        print(f"Error at step {step_num}: {e}")
        break

print(f"Captured {len(states)} game states")

# Embed states and actions as JSON strings
states_json = json.dumps(states, default=str)
actions_json = json.dumps(actions, default=str)

# Create HTML with embedded data
html_template = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Kaggriculture - Animated Game Board</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }
        .container { max-width: 1800px; margin: 0 auto; }
        .header {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }
        .header h1 { color: #667eea; margin-bottom: 5px; }
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .board-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .board-container h2 { color: #667eea; margin-bottom: 15px; text-align: center; }
        .board-info {
            display: flex;
            justify-content: space-around;
            margin-bottom: 15px;
            font-size: 0.9em;
        }
        .info-item { text-align: center; }
        .info-label { color: #666; font-size: 0.8em; }
        .info-value { font-weight: bold; color: #667eea; font-size: 1.2em; }
        .board-grid {
            display: grid;
            grid-template-columns: repeat(10, 40px);
            gap: 2px;
            margin: 0 auto;
            width: fit-content;
        }
        .tile {
            width: 40px;
            height: 40px;
            border: 1px solid #ddd;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.4em;
            background: #f9f9f9;
            border-radius: 3px;
            transition: all 0.2s;
        }
        .tile.farmer { background: #fff3cd; border: 2px solid #ffc107; }
        .action-panel {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .action-panel h2 { color: #667eea; margin-bottom: 15px; }
        .action-step {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 15px;
            border-radius: 8px;
        }
        .step-header {
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }
        .actions-list {
            background: white;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 0.85em;
            max-height: 150px;
            overflow-y: auto;
        }
        .action-item {
            padding: 5px;
            margin-bottom: 3px;
            border-left: 3px solid #667eea;
            padding-left: 10px;
            font-size: 0.75em;
        }
        .player-0 { border-left-color: #2ecc71; }
        .player-1 { border-left-color: #e74c3c; }
        .controls {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        button {
            padding: 10px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover { background: #764ba2; }
        .slider-container {
            flex: 1;
            min-width: 200px;
            display: flex;
            gap: 10px;
            align-items: center;
        }
        input[type="range"] {
            flex: 1;
        }
        .legend {
            background: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
            gap: 10px;
            font-size: 0.9em;
        }
        .legend-item { display: flex; align-items: center; gap: 8px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Kaggriculture - Animated Game Board</h1>
        </div>
        
        <div class="controls">
            <button onclick="goToStep(0)">Start</button>
            <button onclick="prevStep()">Previous</button>
            <button id="playBtn" onclick="togglePlay()">Play</button>
            <button onclick="nextStep()">Next</button>
            <button onclick="goToStep(719)">End</button>
            
            <div class="slider-container">
                <input type="range" id="slider" min="0" max="719" value="0" onchange="goToStep(this.value)">
                <span id="stepLabel">0/720</span>
            </div>
            
            <div>
                Speed: <input type="range" id="speedSlider" min="100" max="2000" value="1000" step="100" onchange="setSpeed(this.value)">
                <span id="speedLabel">1000ms</span>
            </div>
        </div>
        
        <div class="legend">
            <div class="legend-item"><span style="font-size:1.3em">⬜</span>Empty</div>
            <div class="legend-item"><span style="font-size:1.3em">🌾</span>Wheat</div>
            <div class="legend-item"><span style="font-size:1.3em">🥕</span>Carrot</div>
            <div class="legend-item"><span style="font-size:1.3em">🍅</span>Tomato</div>
            <div class="legend-item"><span style="font-size:1.3em">🍈</span>Melon</div>
            <div class="legend-item"><span style="font-size:1.3em">🦆</span>Goose</div>
            <div class="legend-item"><span style="font-size:1.3em">🐄</span>Cow</div>
        </div>
        
        <div class="main-content">
            <div class="board-container">
                <h2>Player 0</h2>
                <div class="board-info">
                    <div class="info-item">
                        <div class="info-label">Money</div>
                        <div class="info-value" id="p0Money">$3000</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Pos</div>
                        <div class="info-value" id="p0Pos">4,4</div>
                    </div>
                </div>
                <div class="board-grid" id="p0Board"></div>
            </div>
            
            <div class="board-container">
                <h2>Player 1</h2>
                <div class="board-info">
                    <div class="info-item">
                        <div class="info-label">Money</div>
                        <div class="info-value" id="p1Money">$3000</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Pos</div>
                        <div class="info-value" id="p1Pos">4,4</div>
                    </div>
                </div>
                <div class="board-grid" id="p1Board"></div>
            </div>
        </div>
        
        <div class="action-panel">
            <h2>Current Step</h2>
            <div class="action-step">
                <div class="step-header" id="stepTitle">Step 0 | Day 0</div>
                <div class="actions-list" id="actionsList">No actions</div>
            </div>
        </div>
    </div>
    
    <script>
        const gameStates = %STATES_JSON%;
        const gameActions = %ACTIONS_JSON%;
        
        let currentStep = 0;
        let isPlaying = false;
        let speed = 1000;
        let playInterval = null;
        
        const actionsByStep = {};
        gameActions.forEach(a => {
            if (!actionsByStep[a.step]) actionsByStep[a.step] = [];
            actionsByStep[a.step].push(a);
        });
        
        function getTileEmoji(tile) {
            if (!tile || tile === null) return "⬜";
            if (tile === "LOCKED") return "🔒";
            if (typeof tile === 'object' && tile.kind) {
                if (tile.kind === 'PLANT') {
                    const crops = {'WHEAT': '🌾', 'CARROT': '🥕', 'TOMATO': '🍅', 'STRAWBERRY': '🍓', 'MELON': '🍈'};
                    return crops[tile.crop] || '🌱';
                }
                if (tile.kind === 'WEED') return "🌿";
                if (tile.kind === 'COOP') return '🦆';
                if (tile.kind === 'PASTURE') return tile.animal === 'COW' ? '🐄' : '🐑';
                if (tile.kind === 'SHED') return '📦';
            }
            return "❓";
        }
        
        function renderBoard(boardData, farmerPos) {
            return boardData.map((row, y) => 
                row.map((tile, x) => {
                    const emoji = getTileEmoji(tile);
                    const isFarmer = x === farmerPos[0] && y === farmerPos[1];
                    return `<div class="tile ${isFarmer ? 'farmer' : ''}">${emoji}</div>`;
                }).join('')
            ).join('');
        }
        
        function update() {
            const state = gameStates[currentStep];
            const stepActions = actionsByStep[currentStep] || [];
            
            // Update UI
            const day = Math.floor(currentStep / 24);
            const hour = currentStep % 24;
            
            document.getElementById('slider').value = currentStep;
            document.getElementById('stepLabel').textContent = currentStep + '/720';
            document.getElementById('stepTitle').textContent = `Step ${currentStep} | Day ${day} | Hour ${hour}`;
            
            document.getElementById('p0Money').textContent = '$' + Math.round(state.money[0]);
            document.getElementById('p1Money').textContent = '$' + Math.round(state.money[1]);
            document.getElementById('p0Pos').textContent = state.farmer_pos[0][0] + ',' + state.farmer_pos[0][1];
            document.getElementById('p1Pos').textContent = state.farmer_pos[1][0] + ',' + state.farmer_pos[1][1];
            
            // Render boards
            document.getElementById('p0Board').innerHTML = renderBoard(state.boards[0], state.farmer_pos[0]);
            document.getElementById('p1Board').innerHTML = renderBoard(state.boards[1], state.farmer_pos[1]);
            
            // Show actions
            let html = '';
            stepActions.forEach(a => {
                html += '<div class="action-item player-' + a.player + '">';
                if (a.farmer_action[0] !== 'PASS') {
                    html += '<b>P' + a.player + ' Farmer:</b> ' + JSON.stringify(a.farmer_action) + '<br>';
                }
                if (a.market_orders.length > 0) {
                    html += '<b>P' + a.player + ' Market:</b> ' + JSON.stringify(a.market_orders);
                }
                html += '</div>';
            });
            if (!html) html = '<div class="action-item">No actions</div>';
            document.getElementById('actionsList').innerHTML = html;
        }
        
        function goToStep(s) {
            currentStep = Math.max(0, Math.min(719, parseInt(s)));
            update();
        }
        
        function nextStep() { if (currentStep < 719) goToStep(currentStep + 1); }
        function prevStep() { if (currentStep > 0) goToStep(currentStep - 1); }
        
        function togglePlay() {
            isPlaying = !isPlaying;
            const btn = document.getElementById('playBtn');
            btn.textContent = isPlaying ? 'Pause' : 'Play';
            
            if (isPlaying) {
                playInterval = setInterval(() => {
                    if (currentStep < 719) {
                        nextStep();
                    } else {
                        isPlaying = false;
                        btn.textContent = 'Play';
                        clearInterval(playInterval);
                    }
                }, speed);
            } else {
                clearInterval(playInterval);
            }
        }
        
        function setSpeed(v) {
            speed = parseInt(v);
            document.getElementById('speedLabel').textContent = speed + 'ms';
            if (isPlaying) {
                clearInterval(playInterval);
                playInterval = setInterval(() => {
                    if (currentStep < 719) nextStep();
                    else { isPlaying = false; clearInterval(playInterval); }
                }, speed);
            }
        }
        
        update();
    </script>
</body>
</html>'''

# Replace placeholders
html_content = html_template.replace('%STATES_JSON%', states_json)
html_content = html_content.replace('%ACTIONS_JSON%', actions_json)

# Write file
with open("game_animated.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Created game_animated.html successfully!")
print(f"Open it in your browser to view the animated board replay")
