"""
Creates an animated game board viewer
Loads game_log.json and reconstructs board state changes
"""
import json
from kaggle_environments import make

def create_animated_board_viewer():
    """Create interactive HTML with animated game board"""
    
    # Load action log
    with open("game_log.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    actions = data['action_log']
    
    # Embed actions as JSON
    actions_json = json.dumps(actions, default=str)
    
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
            obs = result
            states.append(capture_state())
        except Exception as e:
            print(f"Error at step {step_num}: {e}")
            break
    
    # Embed states and actions as JSON
    states_json = json.dumps(states, default=str)
    actions_json = json.dumps(actions, default=str)
    
    html = """
<!DOCTYPE html>
<html>
<head>
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
        .header p { color: #666; }
        
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
        
        .board-container h2 {
            color: #667eea;
            margin-bottom: 15px;
            text-align: center;
        }
        
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
            grid-template-columns: repeat(10, 35px);
            gap: 2px;
            margin: 0 auto;
            width: fit-content;
        }
        
        .tile {
            width: 35px;
            height: 35px;
            border: 1px solid #ddd;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3em;
            background: #f9f9f9;
            border-radius: 3px;
            position: relative;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .tile:hover { background: #f0f0f0; }
        .tile.farmer { background: #fff3cd; }
        .tile.hand { background: #e7f3ff; }
        
        .action-panel {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .action-panel h2 {
            color: #667eea;
            margin-bottom: 15px;
        }
        
        .action-step {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        
        .step-header {
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        
        .actions-list {
            background: white;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 0.85em;
        }
        
        .action-item {
            padding: 5px;
            margin-bottom: 5px;
            border-left: 3px solid #667eea;
            padding-left: 10px;
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
            padding: 12px 25px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: background 0.3s;
        }
        
        button:hover { background: #764ba2; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        
        .slider-container {
            flex: 1;
            min-width: 300px;
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        input[type="range"] {
            flex: 1;
            height: 6px;
            border-radius: 3px;
            background: #667eea;
            outline: none;
            -webkit-appearance: none;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #764ba2;
            cursor: pointer;
        }
        
        input[type="range"]::-moz-range-thumb {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #764ba2;
            cursor: pointer;
            border: none;
        }
        
        .legend {
            background: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            font-size: 0.9em;
        }
        
        .legend-item { display: flex; align-items: center; gap: 8px; }
        .legend-emoji { font-size: 1.3em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Kaggriculture - Animated Game Board</h1>
            <p>Watch the farm develop in real-time</p>
        </div>
        
        <div class="controls">
            <button onclick="goToStep(0)">Start</button>
            <button onclick="prevStep()" id="btnPrev">Previous</button>
            <button onclick="togglePlayPause()" id="btnPlay">Play</button>
            <button onclick="nextStep()" id="btnNext">Next</button>
            <button onclick="goToStep(719)">End</button>
            
            <div class="slider-container">
                <input type="range" id="stepSlider" min="0" max="719" value="0" 
                       oninput="goToStep(this.value)">
                <span id="sliderValue" style="color: white; min-width: 60px;">Step 0/720</span>
            </div>
            
            <div style="color: white; font-weight: bold;">
                Speed: <input type="range" id="speedSlider" min="100" max="2000" value="1000" 
                             step="100" style="width: 80px;" oninput="updateSpeed(this.value)">
                <span id="speedValue">1000ms</span>
            </div>
        </div>
        
        <div class="legend">
            <div class="legend-item"><span class="legend-emoji">⬜</span> <span>Empty</span></div>
            <div class="legend-item"><span class="legend-emoji">🌾</span> <span>Wheat</span></div>
            <div class="legend-item"><span class="legend-emoji">🥕</span> <span>Carrot</span></div>
            <div class="legend-item"><span class="legend-emoji">🍅</span> <span>Tomato</span></div>
            <div class="legend-item"><span class="legend-emoji">🍓</span> <span>Strawberry</span></div>
            <div class="legend-item"><span class="legend-emoji">🍈</span> <span>Melon</span></div>
            <div class="legend-item"><span class="legend-emoji">🦆</span> <span>Goose</span></div>
            <div class="legend-item"><span class="legend-emoji">🐄</span> <span>Cow</span></div>
            <div class="legend-item"><span class="legend-emoji">👨</span> <span>Farmer</span></div>
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
                        <div class="info-label">Farmer Pos</div>
                        <div class="info-value" id="p0Pos">(4,4)</div>
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
                        <div class="info-label">Farmer Pos</div>
                        <div class="info-value" id="p1Pos">(4,4)</div>
                    </div>
                </div>
                <div class="board-grid" id="p1Board"></div>
            </div>
        </div>
        
        <div class="action-panel">
            <h2>Current Step</h2>
            <div class="action-step">
                <div class="step-header" id="stepTitle">Step 0 | Day 0 | Hour 0</div>
                <div class="actions-list" id="actionsList">
                    No actions recorded
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const actions = """ + actions_json + """;
        
        let currentStep = 0;
        let isPlaying = false;
        let playSpeed = 1000;
        let playInterval = null;
        
        // Group actions by step
        const actionsByStep = {};
        actions.forEach(a => {
            if (!actionsByStep[a.step]) actionsByStep[a.step] = [];
            actionsByStep[a.step].push(a);
        });
        
        
        function getTileEmoji(tile) {
            if (tile === null) return "⬜";
            if (tile === "LOCKED") return "🔒";
            if (typeof tile === 'object') {
                if (tile.kind === 'WEED') return "🌿";
                if (tile.kind === 'PLANT') {
                    const crops = {'WHEAT': '🌾', 'CARROT': '🥕', 'TOMATO': '🍅', 
                                  'STRAWBERRY': '🍓', 'MELON': '🍈'};
                    return crops[tile.crop] || '🌱';
                }
                if (tile.kind === 'COOP') return tile.animal ? '🦆' : '🏚️';
                if (tile.kind === 'PASTURE') {
                    return tile.animal ? (tile.animal === 'COW' ? '🐄' : '🐑') : '🚜';
                }
            }
            return "❓";
        }
        
        function renderBoard(boardData, playerId) {
            const html = boardData.map((row, y) => 
                if (!tile || tile === null) return "⬜";
                if (tile === "LOCKED" || (tile.kind && tile.kind === "LOCKED")) return "🔒";
            
                if (typeof tile === 'object' && tile.kind) {
                    const kind = tile.kind;
                    const isF = [x, y][0] === farmer[0] && [x, y][1] === farmer[1];
                    return `<div class="tile ${isF ? 'farmer' : ''}">${emoji}</div>`;
                }).join('')
            ).join('');
                        return crops[tile.crop] || '🌱';
        }
                    if (kind === 'COOP') return '🦆';
                    if (kind === 'PASTURE') return (tile.animal === 'COW') ? '🐄' : '🐑';
                    if (kind === 'SHED') return '🏚️';
            
            // Simulate processing actions to update board state
            // (This is a simplified representation - full board simulation would require replaying all steps)
            
            function renderBoard(boardData, playerId) {
                const farmerPos = playerId === 0 ? states[currentStep].farmer_pos[0] : states[currentStep].farmer_pos[1];
            
                return boardData.map((row, y) => 
                if (a.player === 0) {
                    boardState.p0_money = a.money_before;
                        const isFarmer = x === farmerPos[0] && y === farmerPos[1];
                        return `<div class="tile ${isFarmer ? 'farmer' : ''}">${emoji}</div>`;
            });
                ).join('');
            const day = Math.floor(step / 24);
            const hour = step % 24;
            document.getElementById('stepTitle').textContent = 
                if (currentStep >= states.length) currentStep = states.length - 1;
                if (currentStep < 0) currentStep = 0;
            
                const state = states[currentStep];
                const stepActions = actionsByStep[currentStep] || [];
            document.getElementById('sliderValue').textContent = `Step ${step}/720`;
                // Update step info
                const day = Math.floor(currentStep / 24);
                const hour = currentStep % 24;
                document.getElementById('stepTitle').textContent = 
                    `Step ${currentStep}/719 | Day ${day}/29 | Hour ${hour}/23`;
                document.getElementById('stepSlider').value = currentStep;
                document.getElementById('sliderValue').textContent = `Step ${currentStep}/720`;
            document.getElementById('p1Money').textContent = `$${Math.round(boardState.p1_money)}`;
                // Update money and positions
                document.getElementById('p0Money').textContent = `$${Math.round(state.money[0])}`;
                document.getElementById('p1Money').textContent = `$${Math.round(state.money[1])}`;
                document.getElementById('p0Pos').textContent = `(${state.farmer_pos[0][0]},${state.farmer_pos[0][1]})`;
                document.getElementById('p1Pos').textContent = `(${state.farmer_pos[1][0]},${state.farmer_pos[1][1]})`;
            
                // Render boards
                document.getElementById('p0Board').innerHTML = renderBoard(state.boards[0], 0);
                document.getElementById('p1Board').innerHTML = renderBoard(state.boards[1], 1);
            
                // Display actions
                let actionsHtml = '';
                const p0Actions = stepActions.filter(a => a.player === 0);
                const p1Actions = stepActions.filter(a => a.player === 1);
            
                if (p0Actions.length > 0) {
                    p0Actions.forEach(a => {
                        actionsHtml += `<div class="action-item player-0">`;
                        if (a.farmer_action[0] !== 'PASS') {
                            actionsHtml += `<strong>P0 Farmer:</strong> ${JSON.stringify(a.farmer_action)}<br>`;
                        }
                        if (a.market_orders.length > 0) {
                            actionsHtml += `<strong>P0 Market:</strong> ${JSON.stringify(a.market_orders)}`;
                        }
                        actionsHtml += '</div>';
                    });
            let actionsHtml = '';
            if (stepActions.length === 0) {
                if (p1Actions.length > 0) {
                    p1Actions.forEach(a => {
                        actionsHtml += `<div class="action-item player-1">`;
                        if (a.farmer_action[0] !== 'PASS') {
                            actionsHtml += `<strong>P1 Farmer:</strong> ${JSON.stringify(a.farmer_action)}<br>`;
                        }
                        if (a.market_orders.length > 0) {
                            actionsHtml += `<strong>P1 Market:</strong> ${JSON.stringify(a.market_orders)}`;
                        }
                        actionsHtml += '</div>';
                    });
                }
                    }
                if (actionsHtml === '') {
                    actionsHtml = '<div class="action-item">No actions this step</div>';
                }
                    }
            const btn = document.getElementById('btnPlay');
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
                }, playSpeed);
            } else {
                clearInterval(playInterval);
            }
        }
        
        function updateSpeed(value) {
            playSpeed = parseInt(value);
            document.getElementById('speedValue').textContent = playSpeed + 'ms';
            if (isPlaying) {
                clearInterval(playInterval);
                playInterval = setInterval(() => {
                    if (currentStep < 719) {
                        nextStep();
                    } else {
                        isPlaying = false;
                        document.getElementById('btnPlay').textContent = 'Play';
                        clearInterval(playInterval);
                    }
                }, playSpeed);
            }
        }
        
        console.log('Loaded ' + states.length + ' game states');
        updateDisplay();
    </script>
</body>
</html>
"""
    
    with open("game_animated.html", "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"Created game_animated.html with {len(states)} captured game states")
    return html

if __name__ == "__main__":
    create_animated_board_viewer()
    print("\nOpen game_animated.html in your browser to watch the game play out!")
