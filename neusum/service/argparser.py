import argparse


def parse_arg_interface():
    parser = argparse.ArgumentParser(description='PyTorch Summarization Model')
    # Mode
    parser.add_argument('--dbg', action='store_true', default=False,
                        help="debug mode or not. If dbg, load less data and no pre-training.")
    parser.add_argument('--mode', type=str, default='train', help="train or test")
    # Meta
    parser.add_argument('--data_name', type=str, default='cnn',
                        help='cnn or dailymail or cnn-dailymail; name of the data corpus')
    parser.add_argument('--seed', type=int, default=2019, help='random seed')
    # Dir
    parser.add_argument('--abs_dir_root', action='store', type=str
                        , default='/backup2/jcxu/exComp'
                        )
    parser.add_argument('--abs_board_file', action='store'
                        # , default='/backup2/jcxu/exComp/board.txt', type=str
                        )
    parser.add_argument('--abs_dir_exp', action='store'
                        , default=''
                        )
    parser.add_argument('--abs_dir_log', action='store'
                        # , default='/backup2/jcxu/exComp/log'
                        )
    parser.add_argument('--abs_dir_data', action='store', default='/backup2/jcxu/data/read_ready-grammarTrue-miniTrue')
    parser.add_argument('--pretrain_embedding', action='store')
    # Hyper param about MODEL
    parser.add_argument('--elmo', action='store_true', default=False, help='Use ELMO')
    parser.add_argument('--elmo_num_output_representations', default=1)
    parser.add_argument('--hid_dim', type=int, default=128, help='number of hidden units per layer')
    parser.add_argument('--emb_dim', type=int, default=200, help='size of word embeddings')
    parser.add_argument('--max_dec_step', type=int, default=3, help="Default value for the max decoding steps")
    parser.add_argument('--min_dec_step', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--dropout_emb', type=float, default=0.2)
    parser.add_argument('--span_encoder_type', default='self_attentive')
    parser.add_argument('--nenc_lay', type=int, default=2)
    parser.add_argument('--attn_type', default='general', help='general or dot')

    parser.add_argument('--compression', action='store_true', default=False)
    parser.add_argument('--aggressive_compression', default=-1, action='store',
                        type=int, help="specify the num of deletion: -1=independent classifier,"
                                       "1, 2, 3... means num of deletion.")
    parser.add_argument('--subsentence', default=False, action='store_true', help="Subsentence level selection")
    parser.add_argument('--alpha', action='store', default=1.0, type=float, help="sent loss + alpha * compression loss")
    parser.add_argument('--lazy', action='store_true', default=False, help="Data loading")
    parser.add_argument('--weight_alpha', action='store', default=0.00001, type=float)
    parser.add_argument('--bias_alpha', action='store', default=0.00001, type=float)
    # EDU
    parser.add_argument('--train_fname', type=str, default="train.pkl")
    parser.add_argument('--dev_fname', type=str, default="dev.pkl")
    parser.add_argument('--test_fname', type=str, default="test.pkl")
    parser.add_argument('--single_oracle', action='store_true', default=False)
    parser.add_argument('--fix_edu_num', type=int, default=-1)
    parser.add_argument('--mult_orac_sampling', action='store_true', default=False)
    parser.add_argument('--trim_sent_oracle', action='store_true', default=False)
    # Hyper param about training
    parser.add_argument('--schedule', type=float, default=0.7, help="schedule_ratio_from_ground_truth")  # TODO
    parser.add_argument('--eval_batch_size', type=int, default=20, help='evaluation batch size')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=5.0,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=70,
                        help='upper epoch limit')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--optim', type=str, default='adam', help='sgd or adam')
    # Device
    parser.add_argument('--device_use_cpu', action='store_true', default=False)
    parser.add_argument('--device_cuda_id', action='store', type=int, default=1)
    parser.add_argument('--validation_interval', action='store', type=int, default=500)
    # new
    parser.add_argument('--compress_leadn', default=-1, action='store', type=int)
    parser.add_argument('--dec_avd_trigram_rep', default=False, action='store_true')
    parser.add_argument('--keep_threshold', default=1., action='store', type=float)

    parser.add_argument('--gather', type=str, default='sum', help='mean or sum or attn')
    parser.add_argument('--load_saved_model',type=str)
    """
    # Absolute path
    parser.add_argument('--exp_path', type=str, default='/backup2/jcxu/summa_log', help="Absolute path of experiments")

    # Relative path to root
    parser.add_argument('--root_path', type=str, default='/backup2/jcxu/exComp')
    parser.add_argument('--log_path', type=str, default='log', help="relative location of the log path given root path")

    parser.add_argument('--data_pretrain_emb', type=str,
                        default='data/1-billion-word-language-modeling-benchmark-r13output.word2vec.vec')

    parser.add_argument('--dict_sz', type=int, default=40000, help="The size of the dict")

    parser.add_argument('--num_sample_rollout', type=int, default=20)

    parser.add_argument('--tmp_directory', type=str, default='')
    parser.add_argument('--gold_summary_directory', action='store',
                        type=str, default='data/Baseline-Gold-Models')

    parser.add_argument('--rouge_reweighted', action='store_true', default=False,
                        help="Using Rouge value to reweight the loss")

    # Sequence to Sequence
    parser.add_argument('--enc', type=str, default='lstm', help='lstm or gru or bow')


    """
    args = parser.parse_args()
    return args
