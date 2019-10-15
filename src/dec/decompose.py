import pandas as pd
import numpy as np
import scipy.sparse as sp
import pylab as plt


def compose_operator(deco_dict, df, offset_col='OFFSET'):
    i = 0
    col = []
    data = []
    row = []
    factors = {}
    for k in deco_dict:
        c, t = deco_dict[k]
        x = df[offset_col].values.copy()
        x = x ** t
        data.append(x.tolist())
        col.append((df[c].astype(int).values.copy() + i).tolist())
        row.append(np.arange(len(df), dtype=int).tolist())
        factors[k] = np.arange(df[c].nunique()) + i
        i += df[c].nunique()

    f = np.hstack
    csc = sp.csc_matrix((f(data), (f(row), f(col))))

    print('>>> Compose Operator \n Input size A=({}), y=({})'.format(csc.shape, len(df)))
    print('> Factors are: ', {x: len(factors[x]) for x in factors})
    return csc, factors


def decompose_theta(factors, theta, df, csc, y_col='Time', suffix='', plot_results=True):
    deco_res = {k: theta[factors[k]] for k in factors}

    if plot_results:
        fig, axs = plt.subplots(ncols=len(deco_res), figsize=(20, 3))
        for ax, k in zip(axs.ravel(), deco_res):
            v = deco_res[k]
            if k.endswith('vel0'):
                v = 1 / (np.abs(v) + 1e-16)
            if 'ON' in k:
                ax.plot(v)
            else:
                ax.hist(v, bins=50)
                ax.set_yscale('log')

            ax.set_title(k)

    prefix = y_col + '_F' + str(len(deco_res))
    new_cols = [prefix + suffix]
    df[prefix + suffix] = csc.dot(theta)
    for k in deco_res:
        c = k.split('_')[0]
        v = deco_res[k]
        if k.endswith('vel0'):
            v = 1 / (np.abs(v) + 1e-16)

        tmp = dict(zip(np.sort(df[c].unique()), v))
        df[prefix + '_' + k + suffix] = pd.Series(tmp).loc[df[c].values].values
        new_cols += [prefix + '_' + k + suffix]

    print('>>> Decompose Operator \n Input size A=({}), y=({})'.format(csc.shape, len(df)))
    print('> Factors are: ', list(factors.keys()))
    print('> Columns added to df', new_cols)
    return df