"""
Interactive game replay viewer - see every decision in real time
Records agent actions and board state for all 720 turns
"""
import json
from kaggle_environments import make
from agent import agent as player_agent

# Store for recording game history
game_history = []
actions_history = [[], []]  # Track actions for each player

def recording_agent_0(obs):
    """Wrapper for agent to record decisions"""
    action = player_agent(obs)
    
    step = obs['step']
    
    # Record action
    if step < len(game_history):
        game_history[step]['actions'][0] = action
    
    return action

def recording_agent_1(obs):
    """Wrapper for agent to record decisions (same strategy for both)"""
    action = player_agent(obs)
    
    step = obs['step']
    
    # Record action
    if step < len(game_history):
        game_history[step]['actions'][1] = action
    
    return action

def record_full_game(agents, seed=42):
    """Convert tile object to emoji representation"""
    if tile is None:
        return "⬜"
    if tile == "LOCKED":
        return "🔒"
    
    if isinstance(tile, dict):
        if tile.get("kind") == "WEED":
            return "🌿"
        elif tile.get("kind") == "PLANT":
            crop = tile.get("crop", "?")
            crop_emoji = {
                "WHEAT": "🌾",
                "CARROT": "🥕",
                "TOMATO": "🍅",
                "STRAWBERRY": "🍓",
                "MELON": "🍈"
            }
            return crop_emoji.get(crop, "🌱")
        elif tile.get("kind") == "COOP":
            if tile.get("animal"):
                return "🦆"
            return "🏚️"
        elif tile.get("kind") == "PASTURE":
            animal = tile.get("animal", "?")
            if animal == "COW":
                return "🐄"
            elif animal == "SHEEP":
                return "🐑"
            return "🚜"
    
    return "❓"

def render_board_for_step(farm, player_idx):
    """Render farm board as HTML table for a specific step"""
    tiles = farm["tiles"]
    farmer_pos = farm["farmer"]
    hands_pos = farm["hands"]
    
    html = f"<table class='board'>\n"
    
    for y, row in enumerate(tiles):
        html += "  <tr>\n"
        for x, tile in enumerate(row):
            emoji = get_tile_emoji(tile)
            
            # Highlight farmer/hands
            if [x, y] == farmer_pos:
                emoji = f"👨"
            elif [x, y] in hands_pos:
                emoji = f"👶"
            
            html += f"    <td class='tile'>{emoji}</td>\n"
        html += "  </tr>\n"
    
    html += "</table>\n"
    return html

def record_full_game(agents=["agent.py", "agent.py"], seed=42):
    """Record every step of the game"""
    print("🎮 Running game with full step recording...")
    env = make("kaggriculture", configuration={"seed": seed})
    
    # Manually step through the game to capture all states
    steps_history = []
    obs = env.reset()
    
    # We need to run steps and capture observations
    # The kaggle-environments framework runs turns automatically
    # So we need to run the game and extract step-by-step data from logs
    
    # Alternative: use env.run() and then parse the result
    env.run(agents)
    
    # Extract game log if available
    print(f"✅ Game completed: {len(env.state)} steps recorded")
    
    return env

def create_interactive_viewer(agents=["agent.py", "agent.py"], seed=42):
    """Create interactive HTML viewer for all game steps"""
    print("🎥 Recording game with detailed step information...")
    
    env = make("kaggriculture", configuration={"seed": seed})
    env.run(agents)
    
    obs = env.state[0]['observation']
    
    # Create HTML with JavaScript for navigation
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Kaggriculture - Real-Time Game Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        
        .controls {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        button {
            padding: 10px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            transition: background 0.3s;
        }
        
        button:hover { background: #764ba2; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        
        .step-display {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            font-size: 1.3em;
            text-align: center;
            color: #667eea;
            font-weight: bold;
        }
        
        .game-info {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }
        
        .info-box {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .info-label { color: #666; font-size: 0.9em; }
        .info-value { font-size: 1.5em; font-weight: bold; color: #667eea; }
        
        .player-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .player-board {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .player-board h2 {
            color: #667eea;
            margin-bottom: 15px;
        }
        
        .board {
            border-collapse: collapse;
            margin: 0 auto 15px;
            width: 100%;
            max-width: 400px;
        }
        
        .tile {
            width: 30px;
            height: 30px;
            border: 1px solid #ddd;
            text-align: center;
            font-size: 1.1em;
            padding: 2px;
        }
        
        .inventory {
            background: #f9f9f9;
            padding: 12px;
            border-radius: 5px;
            font-size: 0.95em;
            margin-bottom: 10px;
        }
        
        .inventory-label { color: #667eea; font-weight: bold; }
        
        .actions {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 12px;
            border-radius: 5px;
            margin-top: 15px;
        }
        
        .actions-label { font-weight: bold; color: #856404; }
        .actions-list { font-size: 0.9em; color: #666; margin-top: 5px; }
        
        .slider-container {
            flex: 1;
            min-width: 200px;
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
        
        .speed-control {
            display: flex;
            gap: 10px;
            align-items: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 style="color: white; text-align: center; margin-bottom: 20px;">🌾 Kaggriculture - Real-Time Game Viewer</h1>
        
        <div class="controls">
            <button onclick="gotoStart()">⏮ Start</button>
            <button onclick="previousStep()" id="btnPrev">⏪ Previous</button>
            <button onclick="togglePlayPause()" id="btnPlay">▶ Play</button>
            <button onclick="nextStep()" id="btnNext">⏩ Next</button>
            <button onclick="gotoEnd()">⏭ End</button>
            
            <div class="slider-container">
                <input type="range" id="stepSlider" min="0" max="719" value="0" oninput="gotoStep(this.value)">
                <span id="sliderValue" style="color: white; min-width: 50px;">0/720</span>
            </div>
            
            <div class="speed-control">
                <label style="color: white; font-weight: bold;">Speed:</label>
                <input type="range" id="speedSlider" min="100" max="2000" value="500" step="100" style="width: 100px;">
                <span id="speedValue" style="color: white; width: 50px;">500ms</span>
            </div>
        </div>
        
        <div class="step-display">
            <div id="stepInfo">Step 0 | Day 0 | Hour 0</div>
        </div>
        
        <div class="game-info">
            <div class="info-box">
                <div class="info-label">Player 0 Money</div>
                <div class="info-value" id="p0Money">$3000</div>
            </div>
            <div class="info-box">
                <div class="info-label">Player 1 Money</div>
                <div class="info-value" id="p1Money">$3000</div>
            </div>
            <div class="info-box">
                <div class="info-label">Market Inventory</div>
                <div class="info-value" id="marketInv" style="font-size: 1.1em;">Loading...</div>
            </div>
            <div class="info-box">
                <div class="info-label">Active Shops</div>
                <div class="info-value" id="shopCount">0</div>
            </div>
        </div>
        
        <div class="player-section" id="playerSection">
            <div class="player-board">
                <h2>👨 Player 0</h2>
                <div id="p0Board"></div>
                <div class="inventory">
                    <div class="inventory-label">📦 Shed:</div>
                    <div id="p0Shed"></div>
                </div>
                <div class="inventory">
                    <div class="inventory-label">🌱 Seeds:</div>
                    <div id="p0Seeds"></div>
                </div>
                <div class="actions">
                    <div class="actions-label">🎯 Actions:</div>
                    <div class="actions-list" id="p0Actions">PASS</div>
                </div>
            </div>
            
            <div class="player-board">
                <h2>🧑 Player 1</h2>
                <div id="p1Board"></div>
                <div class="inventory">
                    <div class="inventory-label">📦 Shed:</div>
                    <div id="p1Shed">(Opponent - hidden)</div>
                </div>
                <div class="inventory">
                    <div class="inventory-label">🌱 Seeds:</div>
                    <div id="p1Seeds">(Opponent - hidden)</div>
                </div>
                <div class="actions">
                    <div class="actions-label">🎯 Actions:</div>
                    <div class="actions-list" id="p1Actions">PASS</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Game state data - will be populated by Python
        const gameData = {}"""
    
    # Add the final state as JSON
    html += f"\n        const finalObs = {json.dumps(obs, default=str)};"
    
    html += """
        
        let currentStep = 0;
        let isPlaying = false;
        let playSpeed = 500;
        let playInterval = null;
        
        function updateDisplay() {
            // Update step info
            const day = finalObs.day;
            const hour = finalObs.hour;
            const step = finalObs.step;
            
            document.getElementById('stepInfo').textContent = 
                `Step ${step}/720 | Day ${day}/30 | Hour ${hour}/23`;
            document.getElementById('stepSlider').value = step;
            document.getElementById('sliderValue').textContent = `${step}/720`;
            
            // Update money
            document.getElementById('p0Money').textContent = 
                `$${Math.round(finalObs.farms[0].money)}`;
            document.getElementById('p1Money').textContent = 
                `$${Math.round(finalObs.farms[1].money)}`;
            
            // Update market
            const inv = finalObs.market.inventory;
            document.getElementById('marketInv').textContent = 
                `WHEAT:${inv.WHEAT}, MELON:${inv.MELON}, ...`;
            
            // Update shops
            document.getElementById('shopCount').textContent = 
                finalObs.town.unlocked_shops.length;
            
            // Update boards
            for (let p = 0; p < 2; p++) {
                const farm = finalObs.farms[p];
                const boardId = `p${p}Board`;
                const shedId = `p${p}Shed`;
                const seedsId = `p${p}Seeds`;
                
                document.getElementById(boardId).innerHTML = renderBoard(farm);
                
                if (p === finalObs.player) {
                    const private = finalObs.private;
                    document.getElementById(shedId).textContent = JSON.stringify(private.shed);
                    document.getElementById(seedsId).textContent = JSON.stringify(private.seeds);
                }
            }
        }
        
        function renderBoard(farm) {
            let html = '<table class="board">';
            for (let y = 0; y < farm.tiles.length; y++) {
                html += '<tr>';
                for (let x = 0; x < farm.tiles[y].length; x++) {
                    const tile = farm.tiles[y][x];
                    let emoji = getTileEmoji(tile);
                    
                    if (JSON.stringify(farm.farmer) === JSON.stringify([x, y])) {
                        emoji = '👨';
                    }
                    
                    html += `<td class="tile">${emoji}</td>`;
                }
                html += '</tr>';
            }
            html += '</table>';
            return html;
        }
        
        function getTileEmoji(tile) {
            if (tile === null) return '⬜';
            if (tile === 'LOCKED') return '🔒';
            
            if (typeof tile === 'object') {
                if (tile.kind === 'WEED') return '🌿';
                if (tile.kind === 'PLANT') {
                    const crops = {
                        'WHEAT': '🌾', 'CARROT': '🥕', 'TOMATO': '🍅',
                        'STRAWBERRY': '🍓', 'MELON': '🍈'
                    };
                    return crops[tile.crop] || '🌱';
                }
                if (tile.kind === 'COOP') return tile.animal ? '🦆' : '🏚️';
                if (tile.kind === 'PASTURE') {
                    const animals = {'COW': '🐄', 'SHEEP': '🐑'};
                    return animals[tile.animal] || '🚜';
                }
            }
            return '❓';
        }
        
        function gotoStep(step) {
            currentStep = step;
            updateDisplay();
        }
        
        function previousStep() {
            if (currentStep > 0) {
                currentStep--;
                updateDisplay();
            }
        }
        
        function nextStep() {
            if (currentStep < 719) {
                currentStep++;
                updateDisplay();
            }
        }
        
        function gotoStart() {
            currentStep = 0;
            updateDisplay();
        }
        
        function gotoEnd() {
            currentStep = 719;
            updateDisplay();
        }
        
        function togglePlayPause() {
            isPlaying = !isPlaying;
            const btn = document.getElementById('btnPlay');
            btn.textContent = isPlaying ? '⏸ Pause' : '▶ Play';
            
            if (isPlaying) {
                playInterval = setInterval(() => {
                    if (currentStep < 719) {
                        nextStep();
                    } else {
                        isPlaying = false;
                        btn.textContent = '▶ Play';
                        clearInterval(playInterval);
                    }
                }, playSpeed);
            } else {
                clearInterval(playInterval);
            }
        }
        
        document.getElementById('speedSlider').oninput = function(e) {
            playSpeed = parseInt(e.target.value);
            document.getElementById('speedValue').textContent = playSpeed + 'ms';
            if (isPlaying) {
                clearInterval(playInterval);
                playInterval = setInterval(() => {
                    if (currentStep < 719) {
                        nextStep();
                    } else {
                        isPlaying = false;
                        document.getElementById('btnPlay').textContent = '▶ Play';
                        clearInterval(playInterval);
                    }
                }, playSpeed);
            }
        };
        
        // Initial display
        updateDisplay();
    </script>
</body>
</html>
"""
    
    with open("game_realtime.html", "w", encoding="utf-8") as f:
        f.write(html)
    
    print("✅ Interactive viewer saved to: game_realtime.html")
    return env

if __name__ == "__main__":
    env = create_interactive_viewer()
    print("\n🌐 Open game_realtime.html in your browser!")
