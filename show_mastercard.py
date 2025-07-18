#!/usr/bin/env python
import os
import sys
import argparse
import pickle
import numpy as np
from douzero.env.env import is_mastercard

def convert_to_readable_cards(cards):
    """Convert card values to human-readable form, marking mastercards with *."""
    card_names = []
    for card in cards:
        if card == 11:
            name = "J"
        elif card == 12:
            name = "Q"
        elif card == 13:
            name = "K"
        elif card == 14:
            name = "A"
        elif card == 15:
            name = "2"
        elif card == 20:
            name = "SJ"  # Small Joker
        elif card == 30:
            name = "BJ"  # Big Joker
        else:
            name = str(card)
        
        # Mark mastercards with * (jokers are never mastercards)
        if is_mastercard(card):
            name += "*"
        
        card_names.append(name)
    
    return card_names

def patch_game_env_step():
    """Patch the GameEnv.step method to show readable cards."""
    from douzero.env.game import GameEnv
    original_step = GameEnv.step
    
    def new_step(self, action=None):
        result = original_step(self, action)
        
        # After the original step, print readable card info
        if action and action != 'pass':
            readable_cards = convert_to_readable_cards(action)
            print(f"  → {self.acting_player_position} played: {readable_cards}")
            
            # Identify mastercards used as substitutes
            if any(is_mastercard(c) for c in action):
                print("    (Used mastercards)")
        
        return result
    
    # Replace the original method
    GameEnv.step = new_step

def patch_game_env_reset():
    """Patch the GameEnv reset to print information about mastercards."""
    from douzero.env.game import GameEnv
    original_card_play_init = GameEnv.card_play_init
    
    def new_card_play_init(self, card_play_data):
        result = original_card_play_init(self, card_play_data)
        print("card_play_data",card_play_data)
        
        # After initialization, print mastercard info
        if 'mastercard_list' in card_play_data:
            mastercard_list = card_play_data['mastercard_list']
            readable_mastercards = convert_to_readable_cards(mastercard_list)
            print(f"\nMastercards for this game: {readable_mastercards}")
            print("Note: Jokers (SJ, BJ) are never mastercards\n")
        
        return result
    
    # Replace the original method
    GameEnv.card_play_init = new_card_play_init

def main():
    ite = 141062400
    parser = argparse.ArgumentParser(description='Show detailed mastercard gameplay for DouZero')
    parser.add_argument('--landlord', type=str,
            default=f'douzero_checkpoints/douzero/landlord_weights_{ite}.ckpt')
    parser.add_argument('--landlord_up', type=str,
            default=f'douzero_checkpoints/douzero/landlord_up_weights_{ite}.ckpt')
    parser.add_argument('--landlord_down', type=str,
            default=f'douzero_checkpoints/douzero/landlord_down_weights_{ite}.ckpt')
    parser.add_argument('--eval_data', type=str, default='eval_data.pkl', help='Eval data path')
    parser.add_argument('--num_games', type=int, default=20, help='Number of games to display')
    
    args = parser.parse_args()
    
    # Apply patches to show readable cards
    patch_game_env_step()
    patch_game_env_reset()
    
    # Set CUDA device if using GPU
    cuda_device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    print(f"Using CUDA device: {cuda_device}")
    
    # Prepare a smaller eval dataset with only num_games games
    with open(args.eval_data, 'rb') as f:
        data = pickle.load(f)
    
    # Take just the requested number of games
    smaller_data = data[:args.num_games]
    temp_eval_data = 'temp_eval_data.pkl'
    with open(temp_eval_data, 'wb') as f:
        pickle.dump(smaller_data, f)
    
    try:
        # Import this only after patching
        from douzero.evaluation.simulation import evaluate
        
        # Run evaluation with verbose output
        evaluate(
            args.landlord,
            args.landlord_up,
            args.landlord_down,
            temp_eval_data,
            num_workers=1,  # Always use 1 worker for verbose output
            verbose=True,
            show_action=True
        )
    finally:
        # Clean up temporary file
        if os.path.exists(temp_eval_data):
            os.remove(temp_eval_data)

if __name__ == '__main__':
    main() 