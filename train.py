import os

from douzero.dmc import parser, train

if __name__ == '__main__':
    flags = parser.parse_args()
    flags.num_actors = 1
    flags.num_threads = 128
    flags.load_model = False
    flags.batch_size = 32
    flags.savedir = "oracle_reward"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    train(flags)
