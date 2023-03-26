import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument(
        '--DS', dest='DS', help='Dataset, see website: https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets', default='KKI')
    parser.add_argument('--local', dest='local', action='store_const',
                        const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const',
                        const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const',
                        const=True, default=False)

    parser.add_argument('--lr', dest='lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=2,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=10,
                        help='')

    return parser.parse_args()
