import sys
import pandas as pd

def go2():
    main_df = pd.read_csv('backup2/log')
    DSs = main_df.DS.unique()
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
    main_df = pd.read_csv('log')
    DSs = main_df.DS.unique()
    for DS in DSs:
        df = main_df[main_df.DS == DS]
        print('=================')
        print(DS)
        print('=================')
        for concat in ['concat', 'noconcat']:
            for used_info in ['nor', 'all']:
                tmpddf = df[(df.concat == concat) & (df.used_info == used_info)]
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
                        print(gc, concat, used_info, tpe, mx, mx_std)
                        print(gc, concat, used_info, 'Random-Init', tmpdf['_1'].mean(), tmpdf['_1'].std())

if __name__ == '__main__':
    go()
