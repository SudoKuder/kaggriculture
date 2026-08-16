"""
Visual game viewer - shows board state and statistics
Run: python view_game_visual.py
Then open game_board.html in your browser
"""
import json
from pathlib import Path
from kaggle_environments import make

def get_tile_emoji(tile):
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

def render_board(farm, player_idx):
    """Render farm board as HTML table"""
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
                emoji = f"👨 {emoji}"
            elif [x, y] in hands_pos:
                emoji = f"👶 {emoji}"
            
            html += f"    <td class='tile'>{emoji}</td>\n"
        html += "  </tr>\n"
    
    html += "</table>\n"
    return html

def create_visual_replay(agents=["agent.py", "agent.py"], seed=42):
    """Create visual HTML replay of game"""
    print("Running simulation...")
    env = make("kaggriculture", configuration={"seed": seed})
    env.run(agents)
    
    obs = env.state[0]['observation']
    
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Kaggriculture - Game Board</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: Arial, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 30px;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { text-align: center; color: white; margin-bottom: 30px; font-size: 2.5em; }
        
        .game-stats {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
        }
        
        .stat-box {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-value { font-size: 2em; font-weight: bold; color: #667eea; }
        .stat-label { color: #666; font-size: 0.9em; margin-top: 5px; }
        
        .player-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
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
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .player-board.player-0 h2::before { content: "👨 "; }
        .player-board.player-1 h2::before { content: "🧑 "; }
        
        .board {
            border-collapse: collapse;
            margin: 0 auto;
            width: 100%;
        }
        
        .tile {
            width: 30px;
            height: 30px;
            border: 1px solid #ddd;
            text-align: center;
            font-size: 1.2em;
            padding: 2px;
        }
        
        .tile:hover {
            background: #f0f0f0;
            transform: scale(1.1);
        }
        
        .inventory {
            margin-top: 15px;
            padding: 10px;
            background: #f9f9f9;
            border-radius: 5px;
            font-size: 0.9em;
        }
        
        .inventory h4 { color: #667eea; margin-bottom: 8px; }
        
        .emoji-legend {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }
        
        .legend-item { display: flex; align-items: center; gap: 10px; }
        .legend-emoji { font-size: 1.5em; }
        
        .market-info {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-top: 30px;
        }
        
        .market-info h2 { color: #667eea; margin-bottom: 15px; }
        
        .price-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
        }
        
        .price-item {
            background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
            padding: 12px;
            border-radius: 6px;
            text-align: center;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌾 Kaggriculture Game Board</h1>
"""
    
    # Game stats
    p0_money = obs['farms'][0]['money']
    p1_money = obs['farms'][1]['money']
    
    html += f"""
        <div class="game-stats">
            <div class="stat-box">
                <div class="stat-value">{obs['step']}/720</div>
                <div class="stat-label">Turn Progress</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">Day {obs['day']}</div>
                <div class="stat-label">Current Day</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">Hour {obs['hour']}/23</div>
                <div class="stat-label">Hour of Day</div>
            </div>
        </div>
"""
    
    # Player boards
    html += """
        <div class="player-section">
"""
    
    for player_idx in [0, 1]:
        farm = obs['farms'][player_idx]
        money = farm['money']
        html += f"""
            <div class="player-board player-{player_idx}">
                <h2>Player {player_idx}</h2>
                <div style="color: #27ae60; font-size: 1.3em; margin-bottom: 10px;"><strong>${money:.0f}</strong></div>
                <div style="font-size: 0.9em; color: #666; margin-bottom: 10px;">
                    Unlocked: {', '.join(farm['unlocked_quadrants'])}
                </div>
"""
        
        html += render_board(farm, player_idx)
        html += """
                <div class="inventory">
"""
        
        if obs['player'] == player_idx:
            private = obs['private']
            html += f"<h4>📦 Shed: {private['shed']}</h4>"
            html += f"<h4>🌱 Seeds: {private['seeds']}</h4>"
        else:
            html += "<p><em>Private info hidden (opponent)</em></p>"
        
        html += """
                </div>
            </div>
"""
    
    html += """
        </div>
        
        <div class="emoji-legend">
            <div class="legend-item"><span class="legend-emoji">⬜</span> <span>Empty</span></div>
            <div class="legend-item"><span class="legend-emoji">🔒</span> <span>Locked Land</span></div>
            <div class="legend-item"><span class="legend-emoji">🌾</span> <span>Wheat</span></div>
            <div class="legend-item"><span class="legend-emoji">🥕</span> <span>Carrot</span></div>
            <div class="legend-item"><span class="legend-emoji">🍅</span> <span>Tomato</span></div>
            <div class="legend-item"><span class="legend-emoji">🍓</span> <span>Strawberry</span></div>
            <div class="legend-item"><span class="legend-emoji">🍈</span> <span>Melon</span></div>
            <div class="legend-item"><span class="legend-emoji">🦆</span> <span>Goose Coop</span></div>
            <div class="legend-item"><span class="legend-emoji">🐄</span> <span>Cow</span></div>
            <div class="legend-item"><span class="legend-emoji">🐑</span> <span>Sheep</span></div>
            <div class="legend-item"><span class="legend-emoji">🌿</span> <span>Weed</span></div>
            <div class="legend-item"><span class="legend-emoji">👨</span> <span>Farmer</span></div>
            <div class="legend-item"><span class="legend-emoji">👶</span> <span>Farm Hand</span></div>
        </div>
"""
    
    # Market info
    html += f"""
        <div class="market-info">
            <h2>💰 Market</h2>
            <p><strong>Inventory:</strong> {obs['market']['inventory']}</p>
            <div class="price-grid">
"""
    
    for product, price in obs['market']['prices'].items():
        html += f"<div class='price-item'>{product}: ${price}</div>\n"
    
    html += """
            </div>
        </div>
        
        <div class="market-info" style="margin-top: 20px;">
            <h2>🏘️ Town</h2>
            <p><strong>Active Shops:</strong> {}</p>
        </div>
    </div>
</body>
</html>
""".format(', '.join(obs['town']['unlocked_shops']) if obs['town']['unlocked_shops'] else 'None yet')
    
    with open("game_board.html", "w", encoding="utf-8") as f:
        f.write(html)
    
    print("✅ Game board saved to: game_board.html")
    print(f"📊 Final Score - Player 0: ${p0_money:.0f} | Player 1: ${p1_money:.0f}")
    
    return html

if __name__ == "__main__":
    create_visual_replay()
    print("\n🌐 Open game_board.html in your browser to view the game!")
