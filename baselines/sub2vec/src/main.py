import argparse


from structural import structural_embedding
from neighborhood import neighborhood_embedding

def main():
    parser = argparse.ArgumentParser(description="sub2vec.")
    parser.add_argument('--input', nargs='?', required=True, help='Input directory')
    parser.add_argument('--property', default='n', choices=['n', 's'], required=True, help='Type of subgraph property to presernve. For neighborhood property add " --property n" and for the structural property " --property s" ')

    parser.add_argument('--walkLength', default=100000, type=int, help='length of random walk on each subgraph')

    # parser.add_argument('--output', required=True, help='Output representation file')

    parser.add_argument('--d', default=300, type=int, help='dimension of learned feautures for each subgraph.')
    
    parser.add_argument('--iter', default=20, type=int, help= 'training iterations')

    parser.add_argument('--windowSize', default=2, type=int,
                      help='Window size of the model.')
    
    parser.add_argument('--p', default=0.5, type=float,
                      help='meta parameter.')

    parser.add_argument('--model', default='dm', choices=['dbon', 'dm'],
                      help='models for learninig vectors SV-DM (dm) or SV-DBON (dbon).')
                      
    args = parser.parse_args()
    from preprocess import preprocess
    print('start preprocessing ..')
    preprocess(args.input)
    
    if args.property == 's':
        structural_embedding(args)
    else:
        neighborhood_embedding(args)
    


if __name__=='__main__':
    main()
