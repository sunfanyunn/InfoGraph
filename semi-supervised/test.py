import pandas as pd
df = pd.read_csv('log')
for target in range(12):
    tmpdf = df[df.target == target]

    sup = tmpdf[(tmpdf.use_unsup_loss == False) & (tmpdf.separate_encoder == False)]
    print(sup)
    sup = sup.loc[sup['val_mae'].idxmin()]['test_mae']
    print(sup)

    # sup_decay = tmpdf[(tmpdf.use_unsup_loss == False) & (tmpdf.separate_encoder == False) & (tmpdf.weight_decay == 1e-4)]['test_mae'].mean()

    tmpddf = tmpdf[(tmpdf.use_unsup_loss == True) & (tmpdf.separate_encoder == True)]
    print(tmpddf)
    unsup = tmpddf.loc[tmpddf['val_mae'].idxmin()]['test_mae']
    print(unsup)

    input()
