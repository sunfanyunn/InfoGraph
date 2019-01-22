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
    main_df = pd.read_csv(sys.argv[1])
    DSs = main_df.DS.unique()
    try:
        if sys.argv[2] in DSs:
            DSs = [sys.argv[2]]
    except:
        pass
    for DS in DSs:
        df = main_df[main_df.DS == DS]
        print('=================')
        print(DS)
        print('=================')
        for extend in list(df.extend.unique()):
            for concat in ['concat', 'noconcat']:
                for used_info in ['nor', 'all']:
                    tmpdddf = df[(df.concat == concat) & (df.extend == extend) &  (df.used_info == used_info)]
                    for lr in tmpdddf.lr.unique():
                        tmpddf = tmpdddf[tmpdddf.lr == lr]
                        gcs = tmpddf.gc.unique()
                        types = tmpddf.type.unique()
                        for gc in gcs:
                            for tpe in types:
                                tmpdf = tmpddf[(tmpddf.gc == gc) & (tmpddf.type == tpe)]

                                mx, mx_std, num = 0, 0, 0
                                for i in range(11):
                                    # print(tmpdf[str(i)].mean())
                                    if tmpdf[str(i)].mean() > mx:
                                        mx, mx_std, num = tmpdf[str(i)].mean(), tmpdf[str(i)].std(), tmpdf[str(i)].shape[0]
                                # if tpe == 'global':
                                    # tpe = DS + '-' + tpe
                                print(extend, gc, lr, used_info, tpe, mx, mx_std, num)
                                print(extend, gc, lr, used_info, 'Random-Init', tmpdf['_1'].mean(), tmpdf['_1'].std())

if __name__ == '__main__':
    go()
