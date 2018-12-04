import matplotlib.pyplot as plt
import sys
import pandas as pd

if __name__ == '__main__':
    # df = pd.read_csv(sys.argv[1])
    df = pd.read_csv('log')
    DS = df['DS'].unique()

    for ds in DS:
        for tpe in ['nor', 'all']:
            tmpdf = df[(df.DS == ds) & (df.type == tpe)]
            gcs = sorted(tmpdf['num_gc_layers'].unique())
            mx = []
            mx_std = []
            mn = []
            mn_std = []
            for gc in gcs:
                mx.append(tmpdf[tmpdf.num_gc_layers == gc]['max'].mean())
                mx_std.append(tmpdf[tmpdf.num_gc_layers == gc]['max'].std())
                mn.append(tmpdf[tmpdf.num_gc_layers == gc]['mean'].mean())
                mn_std.append(tmpdf[tmpdf.num_gc_layers == gc]['mean'].std())
            print(ds, tpe)
            print(gcs)
            print(' '.join(['{:.2f}±{:.2f}'.format(mx[i]*100, mx_std[i]*100) for i in range(len(mx))]))
            print(' '.join(['{:.2f}±{:.2f}'.format(mn[i]*100, mn_std[i]*100) for i in range(len(mx))]))
            print(mn)
            input()

    print(mx)
    plt.plot(epochs, mx, label='max', marker='o')
    plt.plot(epochs, mn, label='mean', marker='v')
    plt.legend()
    plt.savefig('tmp.png')
