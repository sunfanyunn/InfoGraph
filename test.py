import sys
import pandas as pd
DS = sys.argv[1]

df = pd.read_csv('log')
df = df[df.DS == DS]
gcs = df.gc.unique()
types = df.type.unique()
for gc in gcs:
    for tpe in types:
        tmpdf = df[(df.gc == gc) & (df.type == tpe)]
        print(tpe, tmpdf['_1'].mean(), tmpdf['_1'].std())
        for i in range(11):
            print(gc, tpe, i, tmpdf[str(i)].mean(), tmpdf[str(i)].std())
