import os

from douzero.dmc import parser, train

if __name__ == '__main__':
    flags = parser.parse_args()
    flags.num_actors = 9
    flags.num_threads = 192
    flags.load_model = True
    flags.batch_size = 32
    flags.savedir = "oracle_reward"
    flags.use_oracle_reward = True
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    train(flags)
