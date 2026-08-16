"""
Interactive timeline viewer - Shows every action step-by-step
Reads game_log.json and creates an interactive HTML player
"""
import json

def create_timeline_viewer():
    """Load game log and create interactive HTML timeline"""
    
    # Load game log
    with open("game_log.json", "r", encoding="utf-8") as f:
        game_data = json.load(f)
    
    action_log = game_data['action_log']
    final_state = game_data['final_state']
    
    # Convert actions to JavaScript-friendly format
    actions_js = json.dumps(action_log, default=str)
    
    p0_final = f"${final_state['p0_money']:.0f}"
    p1_final = f"${final_state['p1_money']:.0f}"
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Kaggriculture - Action Timeline</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{ max-width: 1200px; margin: 0 auto; }}
        
        .header {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .header h1 {{ color: #667eea; margin-bottom: 10px; font-size: 2.5em; }}
        .header p {{ color: #666; font-size: 1.1em; }}
        
        .controls {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }}
        
        button {{
            padding: 12px 25px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            font-weight: bold;
            transition: background 0.3s;
        }}
        
        button:hover {{ background: #764ba2; }}
        button:disabled {{ background: #ccc; cursor: not-allowed; }}
        
        .slider-container {{
            flex: 1;
            min-width: 300px;
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        
        input[type="range"] {{
            flex: 1;
            height: 6px;
            border-radius: 3px;
            background: #667eea;
            outline: none;
            -webkit-appearance: none;
        }}
        
        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #764ba2;
            cursor: pointer;
        }}
        
        input[type="range"]::-moz-range-thumb {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #764ba2;
            cursor: pointer;
            border: none;
        }}
        
        .step-info {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .step-header {{
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .money-display {{
            font-size: 1.2em;
            color: #27ae60;
            font-weight: bold;
        }}
        
        .actions-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        
        .player-actions {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        
        .player-actions h3 {{
            color: #667eea;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .action-item {{
            background: white;
            padding: 12px;
            border-radius: 5px;
            margin-bottom: 10px;
            font-family: monospace;
            border-left: 3px solid #667eea;
            font-size: 0.95em;
        }}
        
        .action-label {{
            font-weight: bold;
            color: #667eea;
            display: block;
            margin-bottom: 5px;
        }}
        
        .action-value {{
            color: #333;
            word-break: break-all;
        }}
        
        .empty-action {{
            background: #f9f9f9;
            color: #999;
            font-style: italic;
        }}
        
        .timeline-stats {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        
        .stat-box {{
            background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .stat-value {{ font-size: 1.5em; font-weight: bold; color: #d63031; }}
        .stat-label {{ color: #666; font-size: 0.9em; margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Kaggriculture - Action Timeline</h1>
            <p>Watch every decision made in real-time</p>
        </div>
        
        <div class="timeline-stats">
            <div class="stat-box">
                <div class="stat-label">Total Actions</div>
                <div class="stat-value">{len(action_log)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Player 0 Final</div>
                <div class="stat-value">{p0_final}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Player 1 Final</div>
                <div class="stat-value">{p1_final}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Total Turns</div>
                <div class="stat-value">{final_state['total_turns']}/720</div>
            </div>
        </div>
        
        <div class="controls">
            <button onclick="goToAction(0)">Start</button>
            <button onclick="prevAction()" id="btnPrev">Previous</button>
            <button onclick="togglePlayPause()" id="btnPlay">Play</button>
            <button onclick="nextAction()" id="btnNext">Next</button>
            <button onclick="goToAction(actions.length-1)">End</button>
            
            <div class="slider-container">
                <input type="range" id="actionSlider" min="0" max="{len(action_log)-1}" value="0" 
                       oninput="goToAction(this.value)">
                <span id="sliderValue" style="color: white; min-width: 60px;">0/{len(action_log)}</span>
            </div>
            
            <div style="color: white; font-weight: bold;">
                Speed: <input type="range" id="speedSlider" min="100" max="2000" value="500" 
                             step="100" style="width: 80px;" oninput="updateSpeed(this.value)">
                <span id="speedValue">500ms</span>
            </div>
        </div>
        
        <div class="step-info" id="stepInfo">
            <div class="step-header">
                <span id="stepTitle">Loading...</span>
                <span id="moneyDisplay" class="money-display"></span>
            </div>
            <div class="actions-grid" id="actionsDisplay"></div>
        </div>
    </div>
    
    <script>
        const actions = {actions_js};
        let currentActionIndex = 0;
        let isPlaying = false;
        let playSpeed = 500;
        let playInterval = null;
        
        function goToAction(index) {{
            currentActionIndex = Math.min(Math.max(parseInt(index), 0), actions.length - 1);
            document.getElementById('actionSlider').value = currentActionIndex;
            document.getElementById('sliderValue').textContent = 
                (currentActionIndex + 1) + '/' + actions.length;
            updateDisplay();
        }}
        
        function nextAction() {{
            if (currentActionIndex < actions.length - 1) {{
                goToAction(currentActionIndex + 1);
            }}
        }}
        
        function prevAction() {{
            if (currentActionIndex > 0) {{
                goToAction(currentActionIndex - 1);
            }}
        }}
        
        function updateDisplay() {{
            const action = actions[currentActionIndex];
            const nextAction = currentActionIndex < actions.length - 1 ? actions[currentActionIndex + 1] : null;
            
            // Get unique steps to group actions
            let stepStart = action.step;
            let stepEnd = action.step;
            let i = currentActionIndex + 1;
            while (i < actions.length && actions[i].step === action.step) {{
                i++;
            }}
            
            // Step info
            const stepNum = action.step;
            const day = action.day;
            const hour = action.hour;
            
            document.getElementById('stepTitle').textContent = 
                `Step ${{stepNum}}/719 | Day ${{day}}/29 | Hour ${{hour}}/23`;
            
            // Group by player
            const p0Actions = [];
            const p1Actions = [];
            
            let j = currentActionIndex;
            while (j < actions.length && actions[j].step === stepNum) {{
                if (actions[j].player === 0) p0Actions.push(actions[j]);
                else p1Actions.push(actions[j]);
                j++;
            }}
            
            // Update money (show most recent)
            const latestAction = actions[currentActionIndex];
            document.getElementById('moneyDisplay').textContent = 
                `P0: ${{latestAction.money_before.toFixed(0)}}`;
            
            // Display actions
            let html = '<div class="actions-grid">';
            html += displayPlayerActions(p0Actions, 0);
            html += displayPlayerActions(p1Actions, 1);
            html += '</div>';
            
            document.getElementById('actionsDisplay').innerHTML = html;
        }}
        
        function displayPlayerActions(playerActions, playerNum) {{
            const emoji = playerNum === 0 ? '👨' : '🧑';
            let html = '<div class="player-actions">';
            html += `<h3>${{emoji}} Player ${{playerNum}}</h3>`;
            
            if (playerActions.length === 0) {{
                html += '<div class="action-item empty-action">No actions</div>';
            }} else {{
                playerActions.forEach(action => {{
                    html += '<div class="action-item">';
                    
                    if (action.farmer_action && action.farmer_action[0] !== 'PASS') {{
                        html += '<span class="action-label">Farmer:</span>';
                        html += '<span class="action-value">' + JSON.stringify(action.farmer_action) + '</span>';
                    }}
                    
                    if (action.market_orders && action.market_orders.length > 0) {{
                        html += '<span class="action-label">Market:</span>';
                        action.market_orders.forEach(order => {{
                            html += '<span class="action-value">' + JSON.stringify(order) + '</span><br>';
                        }});
                    }}
                    
                    if ((!action.farmer_action || action.farmer_action[0] === 'PASS') && 
                        (!action.market_orders || action.market_orders.length === 0)) {{
                        html += '<span class="action-value" style="color: #999;">PASS</span>';
                    }}
                    
                    html += '</div>';
                }});
            }}
            
            html += '</div>';
            return html;
        }}
        
        function togglePlayPause() {{
            isPlaying = !isPlaying;
            const btn = document.getElementById('btnPlay');
            btn.textContent = isPlaying ? 'Pause' : 'Play';
            
            if (isPlaying) {{
                playInterval = setInterval(() => {{
                    if (currentActionIndex < actions.length - 1) {{
                        nextAction();
                    }} else {{
                        isPlaying = false;
                        btn.textContent = 'Play';
                        clearInterval(playInterval);
                    }}
                }}, playSpeed);
            }} else {{
                clearInterval(playInterval);
            }}
        }}
        
        function updateSpeed(value) {{
            playSpeed = parseInt(value);
            document.getElementById('speedValue').textContent = playSpeed + 'ms';
            
            if (isPlaying) {{
                clearInterval(playInterval);
                playInterval = setInterval(() => {{
                    if (currentActionIndex < actions.length - 1) {{
                        nextAction();
                    }} else {{
                        isPlaying = false;
                        document.getElementById('btnPlay').textContent = 'Play';
                        clearInterval(playInterval);
                    }}
                }}, playSpeed);
            }}
        }}
        
        // Initial display
        updateDisplay();
    </script>
</body>
</html>
"""
    
    with open("game_actions.html", "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"Created game_actions.html with {len(action_log)} actions")
    return action_log

if __name__ == "__main__":
    actions = create_timeline_viewer()
    print(f"\nOpen game_actions.html in your browser to see the action timeline!")
    print(f"Total actions recorded: {len(actions)}")
