from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import logging
import json

# -------------------------------
# GPT-OSS Config
# -------------------------------
SERVER_URL = "http://127.0.0.1:5000/v1/completions"
GPT_MODEL = "Qwen3-14B-Q6_K.gguf"
HEADERS = {"Content-Type": "application/json"}

# -------------------------------
# Flask Setup
# -------------------------------
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

# -------------------------------
# Helpers
# -------------------------------
def build_gto_prompt(data):
    """Turn client game state into a prompt for the model with explicit suit, hand analysis, and multi-player support."""
    hand = data.get("hand", []) or ["unknown", "unknown"]
    community = data.get("community", []) or []
    opponent_actions = data.get("opponent_actions", []) or []
    pot = data.get("pot", 0)
    chips = data.get("chips", 0)
    stage = data.get("stage", "Unknown")
    position = data.get("position", "SB")
    current_bet = data.get("current_bet", 0)
    num_players = data.get("num_players", 2)
    
    # Analyze player's hand
    hand_str = ", ".join(hand) if hand[0] != "unknown" else "unknown"
    hand_type = "unknown"
    hand_strength = "unknown"
    if len(hand) == 2 and hand[0] != "unknown":
        rank1 = hand[0][:-1]
        rank2 = hand[1][:-1]
        suit1 = hand[0][-1]
        suit2 = hand[1][-1]
        is_suited = suit1 == suit2
        hand_type = "suited" if is_suited else "offsuit"
        rank_values = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
                      "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
        r1 = rank_values.get(rank1, 0)
        r2 = rank_values.get(rank2, 0)
        gap = abs(r1 - r2)
        is_connected = gap <= 2
        high_rank = max(r1, r2)
        if high_rank >= 12 or (is_suited and is_connected):
            hand_strength = "strong"
        elif high_rank >= 9 or (is_connected and gap <= 1):
            hand_strength = "medium"
        else:
            hand_strength = "weak"
    
    # Analyze community cards and flush potential
    community_str = ", ".join(community) if community else "none"
    flush_potential = "none"
    if community and len(hand) == 2 and hand[0] != "unknown":
        all_cards = hand + community
        suit_counts = {}
        for card in all_cards:
            suit = card[-1]
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        max_suit_count = max(suit_counts.values(), default=0)
        flush_suit = [s for s, count in suit_counts.items() if count == max_suit_count][0] if max_suit_count >= 3 else None
        if flush_suit and max_suit_count >= 4:
            flush_potential = f"flush draw with {max_suit_count} {flush_suit} cards"

        if flush_suit and (hand[0][-1] == flush_suit or hand[1][-1] == flush_suit):
            flush_potential = f"you have a {flush_potential}"
    
    # Format opponent actions
    opponent_actions_str = "None"
    if opponent_actions:
        formatted_actions = []
        for i, action in enumerate(opponent_actions):
            if action:
                formatted_actions.append(f"Player {i}: {action}")
        opponent_actions_str = "; ".join(formatted_actions) if formatted_actions else "None"
    
    # Build prompt
    prompt = f"""/no_think
You are a GTO (Game Theory Optimal) poker coach for No-Limit Hold'em with {num_players} players.
Game state:
- Your hand: {hand_str} ({hand_type}, {hand_strength} {'connector' if is_connected else 'non-connector'})
- Community cards: {community_str} ({flush_potential})
- Stage: {stage}
- Position: {position}
- Pot size: {pot}
- Your chips: {chips}
- Opponent actions: {opponent_actions_str}
- Current bet to call: {current_bet}
- Number of players: {num_players}
Provide a **short, single-sentence tip** on the best play: bet, call, raise, or fold.
Calculate pot odds and explain how they apply to the current scenario.
Explain the reason for the tip, considering hand strength, position, and number of opponents.
If raising, specify how you determined the raise amount.
Explain any poker jargon used in simple terms.
/no_think"""
    return prompt.strip()

# -------------------------------
# Routes
# -------------------------------
@app.route("/gto_tip", methods=["POST"])
def gto_tip():
    try:
        data = request.get_json()
        logging.debug(f"Received client data: {json.dumps(data, indent=2)}")
        prompt = build_gto_prompt(data)
        logging.debug(f"Built prompt:\n{prompt}")
        payload = {
            "model": GPT_MODEL,
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.3
        }
        logging.debug(f"Sending payload: {json.dumps(payload, indent=2)}")
        response = requests.post(SERVER_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        raw_output = response.json()
        logging.debug(f"Raw server text: {response.text[:500]}...")
        logging.debug(f"Raw model response: {json.dumps(raw_output, indent=2)}")
        tip = raw_output.get("choices", [{}])[0].get("text", "").strip()
        if not tip:
            tip = "No tip generated."
        tags_to_remove = [r"/think", r"</think>", r"/no_think",r"<",r">"]
        for tag in tags_to_remove:
            tip = tip.replace(tag, "").strip()
        logging.debug(f"Cleaned tip: {tip}")
        return jsonify({"gto_tip": tip})
    except Exception as e:
        logging.exception("Error generating GTO tip")
        return jsonify({"tip": "Error generating tip"}), 500

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    app.run(port=8000, debug=True)