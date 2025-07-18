import torch
import numpy as np
from collections import Counter

ACTION_ENCODE_DIM = 57
ENCODE_DIM = 5

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

def check_no_bombs(a):
    sub_a = a[:13*5]  # Take first 65 elements
    reshaped_a = sub_a.reshape(5, -1)
    print("reshaped_a", reshaped_a)

    # Get the 5th row (mastercard label)
    fifth_row = reshaped_a[4,:]

    # Find columns where 5th row is 0 and 1
    zero_cols = (fifth_row == 0)
    one_cols = (fifth_row == 1)
    print("zero_cols", zero_cols)

    zero_cols_sum = reshaped_a[:, zero_cols].sum()
    one_cols_sum = reshaped_a[:, one_cols].sum()
    total_sum = zero_cols_sum + one_cols_sum

    if torch.all(total_sum < 4) and (a[-1] + a[-2] < 2):
        return True
    else:
        return False
    
def is_mastercard(card):
    """Check if a card is a mastercard (jokers are never mastercards)."""
    return card in global_mastercard_values and card < 20  # Jokers (20,30) are never mastercards

def _cards2array(list_cards):
    """
    A utility function that transforms the actions, i.e.,
    A list of integers into card matrix. Here we remove
    the six entries that are always zero and flatten the
    the representations.
    """
    if len(list_cards) == 0:
        return np.zeros(ACTION_ENCODE_DIM, dtype=np.int8)

    matrix = np.zeros([ENCODE_DIM, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(list_cards)

    for card, num_times in counter.items():
            
        if card < 20:
            matrix[0:4, Card2Column[card]] = NumOnes2Array[num_times]
            if is_mastercard(card):  
                matrix[4, Card2Column[card]] = 1
        elif card == 20:
            jokers[0] = 1
        elif card == 30:
            jokers[1] = 1
    return np.concatenate((matrix.flatten('F'), jokers))

# Convert card_play_data to a tensor like variable 'a'
def cards_to_tensor(card_play_data):
        # Prepare a zero array of shape (5, 13)
        arr = np.zeros((5, 13), dtype=int)

        # Helper to fill rows
        def fill_row(cards, row_idx):
            for card in cards:
                if card in Card2Column:
                    arr[row_idx, Card2Column[card]] += 1

        fill_row(card_play_data['landlord'], 0)
        fill_row(card_play_data['landlord_up'], 1)
        fill_row(card_play_data['landlord_down'], 2)
        # mastercard_list and three_landlord_cards can be handled similarly if needed

        # Example: fill mastercard_list in row 4 if needed
        fill_row(card_play_data.get('mastercard_list', []), 4)

        # Encode each cell using NumOnes2Array and flatten
        encoded_rows = []
        for row in arr:
            encoded_row = [NumOnes2Array[min(cell, 4)] for cell in row]
            # encoded_row is a list of arrays of shape (4,), one for each card column (13 total)
            # Stack to shape (4, 13) instead of (13, 4)
            encoded_rows.append(np.stack(encoded_row, axis=1))
            print("encoded_row", np.asarray(encoded_row).shape)
        # encoded_rows = encoded_rows.reshape(-1)
        arr = np.stack(encoded_rows)

        print("card_play_data", arr.shape)

        # Flatten in row-major order (C order)
        flat = arr.reshape(-1)
      
        # Pad to 65*5 if needed (like 'a')
        tensor_a = torch.zeros(65*5, dtype=torch.int)
        tensor_a[:flat.shape[0]] = torch.from_numpy(flat)

        return tensor_a

card_play_data = {
        'landlord':      [3, 3, 3, 4, 5,5, 6, 9, 9, 9, 10, 8, 11, 11, 12, 12, 12, 14, 14, 20],
        'landlord_up':   [3, 4, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 10, 10, 12, 13,17],
        'landlord_down': [4, 4, 5, 7, 9, 10,11, 11, 13, 13, 13, 14, 14, 17, 17, 17,  30],
        'mastercard_list': [ ], 
        'three_landlord_cards': [5,5,20]   
    }
global_mastercard_values = card_play_data['mastercard_list']
tensor_from_data = torch.from_numpy(_cards2array(card_play_data))
print("tensor_from_data:", tensor_from_data)

print("check_no_bombs(a):", check_no_bombs(tensor_from_data))