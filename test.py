import sys
import pandas as pd

def go2():
    main_df = pd.read_csv('backup2/log')
    DSs = main_df.DS.unique()
    if sys.argv[1] in DSs:
        DSs = [sys.argv[1]]
    for DS in DSs:
        df = main_df[main_df.DS == DS]
        print('=================')
        print(DS)
        print('=================')
        for used_info in ['nor', 'all']:
            tmpddf = df[(df.used_info == used_info)]
            gcs = tmpddf.gc.unique()
            types = tmpddf.type.unique()
            for gc in gcs:
                for tpe in types:
                    tmpdf = tmpddf[(tmpddf.gc == gc) & (tmpddf.type == tpe)]

                    mx, mx_std = 0, 0
                    for i in range(11):
                        if tmpdf[str(i)].mean() > mx:
                            mx, mx_std = tmpdf[str(i)].mean(), tmpdf[str(i)].std()
                    # if tpe == 'global':
                        # tpe = DS + '-' + tpe
                    print(gc, used_info, tpe, mx, mx_std)
                    print(gc, used_info, 'Random-Init', tmpdf['_1'].mean(), tmpdf['_1'].std())


def go():
    main_df = pd.read_csv('log_0')
    DSs = main_df.DS.unique()
    if sys.argv[1] in DSs:
        DSs = [sys.argv[1]]
    for DS in DSs:
        df = main_df[main_df.DS == DS]
        print('=================')
        print(DS)
        print('=================')
        for concat in ['concat', 'noconcat']:
            for used_info in ['nor', 'all']:
                tmpdddf = df[(df.concat == concat) & (df.used_info == used_info)]
                for lr in tmpdddf.lr.unique():
                    tmpddf = tmpdddf[tmpdddf.lr == lr]
                    gcs = tmpddf.gc.unique()
                    types = tmpddf.type.unique()
                    for gc in gcs:
                        for tpe in types:
                            tmpdf = tmpddf[(tmpddf.gc == gc) & (tmpddf.type == tpe)]

                            mx, mx_std, num = 0, 0, 0
                            for i in range(11):
                                if tmpdf[str(i)].mean() > mx:
                                    mx, mx_std, num = tmpdf[str(i)].mean(), tmpdf[str(i)].std(), tmpdf[str(i)].shape[0]
                            # if tpe == 'global':
                                # tpe = DS + '-' + tpe
                            print(gc, lr, used_info, tpe, mx, mx_std, num)
                            print(gc, lr, used_info, 'Random-Init', tmpdf['_1'].mean(), tmpdf['_1'].std())

if __name__ == '__main__':
    go()
