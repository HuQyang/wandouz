import multiprocessing as mp
import pickle
from douzero.evaluation.deep_agent import DeepAgent, DeepAgent_Alpha
import torch

# def load_card_play_models(card_play_model_path_dict):
#     players = {}

#     for position in ['landlord', 'landlord_up', 'landlord_down']:
#         if card_play_model_path_dict[position] == 'rlcard':
#             from .rlcard_agent import RLCardAgent
#             players[position] = RLCardAgent(position)
#         elif card_play_model_path_dict[position] == 'random':
#             from .random_agent import RandomAgent
#             players[position] = RandomAgent()
#         else:
#             from .deep_agent import DeepAgent
#             players[position] = DeepAgent(position, card_play_model_path_dict[position])
#     return players

# def mp_simulate(card_play_data_list, card_play_model_path_dict, q):

#     players = load_card_play_models(card_play_model_path_dict)

#     env = GameEnv(players)
#     for idx, card_play_data in enumerate(card_play_data_list):
#         env.card_play_init(card_play_data)
#         while not env.game_over:
#             env.step()
#         env.reset()

#     q.put((env.num_wins['landlord'],
#            env.num_wins['farmer'],
#            env.num_scores['landlord'],
#            env.num_scores['farmer']
#          ))


# def data_allocation_per_worker(card_play_data_list, num_workers):
#     card_play_data_list_each_worker = [[] for k in range(num_workers)]
#     for idx, data in enumerate(card_play_data_list):
#         card_play_data_list_each_worker[idx % num_workers].append(data)

#     return card_play_data_list_each_worker

# def evaluate(landlord, landlord_up, landlord_down, eval_data, num_workers):

#     with open(eval_data, 'rb') as f:
#         card_play_data_list = pickle.load(f)

#     card_play_data_list_each_worker = data_allocation_per_worker(
#         card_play_data_list, num_workers)
#     del card_play_data_list

#     card_play_model_path_dict = {
#         'landlord': landlord,
#         'landlord_up': landlord_up,
#         'landlord_down': landlord_down}

#     num_landlord_wins = 0
#     num_farmer_wins = 0
#     num_landlord_scores = 0
#     num_farmer_scores = 0

#     ctx = mp.get_context('spawn')
#     q = ctx.SimpleQueue()
#     processes = []
#     for card_paly_data in card_play_data_list_each_worker:
#         p = ctx.Process(
#                 target=mp_simulate,
#                 args=(card_paly_data, card_play_model_path_dict, q))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()

#     for i in range(num_workers):
#         result = q.get()
#         num_landlord_wins += result[0]
#         num_farmer_wins += result[1]
#         num_landlord_scores += result[2]
#         num_farmer_scores += result[3]

#     num_total_wins = num_landlord_wins + num_farmer_wins
#     print('WP results:')
#     print('landlord : Farmers - {} : {}'.format(num_landlord_wins / num_total_wins, num_farmer_wins / num_total_wins))
#     print('ADP results:')
#     print('landlord : Farmers - {} : {}'.format(num_landlord_scores / num_total_wins, 2 * num_farmer_scores / num_total_wins)) 


def load_card_play_models(card_play_model_path_dict,device):
    players = {}

    for position in ['landlord', 'landlord_up', 'landlord_down']:
        model_path = card_play_model_path_dict[position]
        if model_path == 'rlcard':
            from .rlcard_agent import RLCardAgent
            players[position] = RLCardAgent(position)
        elif model_path == 'random':
            from .random_agent import RandomAgent
            players[position] = RandomAgent()
        else:
            # if 'weights' in model_path:
            #     players[position] = DeepAgent(position, model_path,device)
            # else:
            #     print("model_path", model_path)
            #     players[position] = DeepAgent_Alpha(position, model_path,device)
            players[position] = DeepAgent(position, model_path,device)
    return players

def mp_simulate(card_play_data_list, card_play_model_path_dict, q, verbose=True,show_action=True,device=torch.device("cuda")):
    # Set resource limits for this process
    # set_process_limits()
    
    # Set up logging output
    import sys
    from io import StringIO
    from douzero.env.env import set_global_mastercards
    
    # If verbose is False, suppress print output from GameEnv
    if not verbose:
        original_stdout = sys.stdout
        sys.stdout = StringIO()  # Redirect stdout to nowhere
    
    players = load_card_play_models(card_play_model_path_dict,device)
    
    if 'weight' in card_play_model_path_dict:
        import douzero.env.game as game
        env = game.GameEnv(players, show_action=show_action)
    else:
        import douzero.env.game_alpha as game_alpha
        env = game_alpha.GameEnv(players, show_action=show_action)
    
    # Using a batch approach to free memory periodically
    batch_size = 20  # Process 20 games before clearing memory

    # print("card_play_data_list",card_play_data_list)

    for i in range(0, len(card_play_data_list), batch_size):
        batch = card_play_data_list[i:min(i+batch_size, len(card_play_data_list))]
        for game_idx, card_play_data in enumerate(batch):
            if verbose:
                print(f"\n===== Starting Game {i + game_idx} =====")
                print(f"Landlord cards: {card_play_data['landlord']}")
                print(f"Landlord up cards: {card_play_data['landlord_up']}")
                print(f"Landlord down cards: {card_play_data['landlord_down']}")
                print(f"Three landlord cards: {card_play_data['three_landlord_cards']}")
                if 'mastercard_list' in card_play_data:
                    print(f"Mastercards: {card_play_data['mastercard_list']}")
                print("======== Game Play Sequence ========")
                # set_global_mastercards(card_play_data['mastercard_list'])
            
            env.card_play_init(card_play_data)
            while not env.game_over:
                env.step()
            
            if verbose:
                print("======== Game End ========\n")
            
            env.reset()
        
        # For Python, suggesting memory cleanup between batches
        import gc
        gc.collect()
    
    # Restore stdout if we redirected it
    if not verbose:
        sys.stdout = original_stdout
    
    q.put((env.num_wins['landlord'],
           env.num_wins['farmer'],
           env.num_scores['landlord'],
           env.num_scores['farmer']
         ))

def data_allocation_per_worker(card_play_data_list, num_workers):
    card_play_data_list_each_worker = [[] for k in range(num_workers)]
    for idx, data in enumerate(card_play_data_list):
        card_play_data_list_each_worker[idx % num_workers].append(data)

    return card_play_data_list_each_worker

def evaluate(landlord, landlord_up, landlord_down, eval_data, num_workers, verbose=False,show_action=True,device=torch.device("cuda")):
    # Cap the number of workers at 120
    max_workers = 120
    num_workers = min(num_workers, max_workers)
    print(f"Using {num_workers} workers for evaluation")
    
    # If verbose mode is on, use only 1 worker to keep output readable
    if verbose:
        num_workers = 1
        print("Verbose mode enabled - using 1 worker for detailed output")

    # Ensure each process gets its own CPU core when possible
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set
        pass

    with open(eval_data, 'rb') as f:
        card_play_data_list = pickle.load(f)

    card_play_data_list_each_worker = data_allocation_per_worker(
        card_play_data_list, num_workers)
    del card_play_data_list

    card_play_model_path_dict = {
        'landlord': landlord,
        'landlord_up': landlord_up,
        'landlord_down': landlord_down}

    num_landlord_wins = 0
    num_farmer_wins = 0
    num_landlord_scores = 0
    num_farmer_scores = 0

    ctx = mp.get_context('spawn')
    q = ctx.SimpleQueue()
    processes = []
    
    for card_paly_data in card_play_data_list_each_worker:
        p = ctx.Process(
                target=mp_simulate,
                args=(card_paly_data, card_play_model_path_dict, q, verbose,show_action,device))
        # Ensure each process uses a reasonable amount of memory
        # This is done by setting a low nice value (higher priority)
        p.daemon = True  # Ensure process terminates when main process does
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for i in range(num_workers):
        result = q.get()
        num_landlord_wins += result[0]
        num_farmer_wins += result[1]
        num_landlord_scores += result[2]
        num_farmer_scores += result[3]

    num_total_wins = num_landlord_wins + num_farmer_wins
    print('WP results:')
    print('landlord : Farmers - {} : {}'.format(num_landlord_wins / num_total_wins, num_farmer_wins / num_total_wins))
    print('ADP results:')
    print('landlord : Farmers - {} : {}'.format(num_landlord_scores / num_total_wins, 2 * num_farmer_scores / num_total_wins)) 
