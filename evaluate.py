import os 
import argparse

from douzero.evaluation.simulation import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dou Dizhu Evaluation')
#     parser.add_argument('--landlord', type=str,
#             default='douzero_checkpoints/douzero/landlord_weights_29257600.ckpt')
#     parser.add_argument('--landlord_up', type=str,
#             default='baselines/sl/landlord_down.ckpt')
#     parser.add_argument('--landlord_down', type=str,
#             default='baselines/sl/landlord_down.ckpt')

    ite1 = 773708800
    ite2 = 1342233600
    ite3 = 1412819200
    ite4 = 1885459200

    alpha_ite1 = 454051200

    # landlord = f'douzero_checkpoints/douzero/landlord_weights_{ite4}.ckpt'
    # landlord_up = f'../beta/douzero_checkpoints/douzero/landlord_up_{alpha_ite1}.ckpt'
    # landlord_down = f'../beta/douzero_checkpoints/douzero/landlord_down_{alpha_ite1}.ckpt'

    landlord = f'../beta/douzero_checkpoints/douzero/landlord_{alpha_ite1}.ckpt'
    landlord_up = f'douzero_checkpoints/douzero/landlord_up_weights_{ite4}.ckpt'
    landlord_down = f'douzero_checkpoints/douzero/landlord_down_weights_{ite4}.ckpt'

    # landlord = f'douzero_checkpoints/douzero/landlord_weights_{ite4}.ckpt'
    # landlord_up = f'douzero_checkpoints/douzero/landlord_up_weights_{ite3}.ckpt'
    # landlord_down = f'douzero_checkpoints/douzero/landlord_down_weights_{ite3}.ckpt'

    landlord = f'../beta/douzero_checkpoints/douzero/landlord_{alpha_ite1}.ckpt'
    landlord_up = f'../beta/douzero_checkpoints/douzero/landlord_up_{alpha_ite1}.ckpt'
    landlord_down = f'../beta/douzero_checkpoints/douzero/landlord_down_{alpha_ite1}.ckpt'

    parser.add_argument('--landlord', type=str, default=landlord)
    parser.add_argument('--landlord_up', type=str, default=landlord_up)
    parser.add_argument('--landlord_down', type=str, default=landlord_down)

#     parser.add_argument('--landlord', type=str,default='random')
#     parser.add_argument('--landlord_up', type=str, default='random')
#     parser.add_argument('--landlord_down', type=str, default='random')

    parser.add_argument('--eval_data', type=str,
            default='eval_data.pkl')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--gpu_device', type=str, default='')
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print(f'landlord: {args.landlord}')
    print(f'landlord_up: {args.landlord_up}')
    print(f'landlord_down: {args.landlord_down}')

    evaluate(args.landlord,
             args.landlord_up,
             args.landlord_down,
             args.eval_data,
             args.num_workers)
