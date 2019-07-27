from IPython.core.display import display
import numpy as np
import pandas as pd
from .read_segy import header_info


PRIME_TIME_COLUMNS = ['SRCX', 'SRCY', 'SRCZ', 'GRPX', 'GRPY', 'GRPZ', 'FB']

src_cols = ['SRCX', 'SRCY', 'SRCZ']
grp_cols = ['GRPX', 'GRPY', 'GRPZ']

src_o_cols = ['SRCX', 'SRCY']
grp_o_cols = ['GRPX', 'GRPY']


def read_prime_times(filename, names=PRIME_TIME_COLUMNS, verbose=True):
    if verbose:
        print('>' * 10, filename, '\n>>> Preview')
        display(
            pd.read_csv(
                filename,
                nrows=3,
                names=names,
                sep='\s+'
            )
        )

    df = pd.read_csv(
        filename,
        #     nrows=3,
        names=['SRCX', 'SRCY', 'SRCZ', 'GRPX', 'GRPY', 'GRPZ', 'FB'],
        sep='\s+'
    )
    src_cols = ['SRCX', 'SRCY', 'SRCZ']
    grp_cols = ['GRPX', 'GRPY', 'GRPZ']

    src_o_cols = ['SRCX', 'SRCY']
    grp_o_cols = ['GRPX', 'GRPY']

    df['OFFSET'] = np.sqrt(
        ((df[src_o_cols].astype(float).values - df[grp_o_cols].astype(float).values) ** 2).sum(axis=1))
    df['RAW_IDX'] = df.index

    sources = df[src_cols].drop_duplicates().reset_index(drop=True)
    sources['SRCID'] = sources.index

    groups = df[grp_cols].drop_duplicates().reset_index(drop=True)
    groups['GRPID'] = groups.index

    df = pd.merge(df, sources, on=src_cols)
    df = pd.merge(df, groups, on=grp_cols)
    df = df.sort_values('RAW_IDX').reset_index(drop=True)
    if verbose:
        print('\n>>> Read complete')
        display(df.head(3))
        header_info(df, name=filename.split('/')[-1])
    return df


COLS_TO_SAVE = ['SRCX', 'SRCY', 'SRCZ', 'GRPX', 'GRPY', 'GRPZ', 'FB']


def save_prime_times(df, filename, columns=COLS_TO_SAVE):
    df[columns].to_csv(filename, index=False, header=False, sep='\t')
