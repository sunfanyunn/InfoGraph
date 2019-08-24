import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--target', dest='target', type=int, default=0,
                        help='')
    parser.add_argument('--train-num', dest='train_num', type=int, default=5000)
    #parser.add_argument('--multi-target', dest='multi_target', type=str, help='')
    parser.add_argument('--use-unsup-loss', dest='use_unsup_loss', action='store_const', const=True, default=False)
    parser.add_argument('--separate-encoder', dest='separate_encoder', action='store_const', const=True, default=False)

    parser.add_argument('--lr', dest='lr', type=float, default=0.01,
            help='Learning rate.')
    parser.add_argument('--lamda', dest='lamda', type=float, default=0.001,
            help='Learning rate.')
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=0,
            help='')

    # parser.add_argument('--', dest='num_gc_layers', type=int, default=5,
            # help='Number of graph convolution layers before each pooling')
    # parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32,
            # help='')

    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    print(args)
