import segyio
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
from obspy.io.segy.core import _read_segy


SEGYIO_HEADER_ITEMS = {
    'FieldRecord':                                      "FFID",
    'TRACE_SEQUENCE_LINE':                              "SEQWL",
    'EnergySourcePoint':                                "SPID",
    'SourceX':                                          "SRCX",
    'SourceY':                                          "SRCY",
    'GroupX':                                           "GRPX",
    'GroupY':                                           "GRPY",

    'offset':                                           "OFFSET",
    'INLINE_3D':                                        "INLINE",
    'CROSSLINE_3D':                                     "XLINE",
}

SEISEE_HEADER_ITEMS = {
    "Trace index in file":                              "IDX",
    "Trace number within field record":                 "TRCFLD",
    "SP - Energy source point number":                  "SPID",
    "CDP ensemble number":                              "CDP",
    "Distance from source point to receiv grp":         "OFFSET",
    "Receiver group elevation":                         "GRPZ",
    "Surface elevation at source":                      "SRCZ",
    "Source X coordinate":                              "SRCX",
    "Source Y coordinate":                              "SRCY",
    "Group X coordinate":                               "GRPX",
    "Group Y coordinate":                               "GRPY",
    "CDP X":                                            "CDPX",
    "CDP Y":                                            "CDPY",
    "Inline Number":                                    "ILINE",
    "Clossline Number":                                 "XLINE",
}


PRIME_TIME_COLUMNS = ['SRCX', 'SRCY', 'SRCZ', 'GRPX', 'GRPY', 'GRPZ', 'FB']

src_cols = ['SRCX', 'SRCY', 'SRCZ']
grp_cols = ['GRPX', 'GRPY', 'GRPZ']

src_o_cols = ['SRCX', 'SRCY']
grp_o_cols = ['GRPX', 'GRPY']


def header_info(df, name=None):
    """
    Print statistics of segy header
    :param df: sgy header DataFrame
    :param name: header name
    :return:
    """
    if not isinstance(name, str):
        name = str(name)

    heading = "{} SGY HEADER: '{:^20}' {}".format('>' * 10, name, '<' * 10)
    print(heading)
    print('df columns: ', np.sort(df.columns.tolist()))
    print('>>> {:15} {:.0f}'.format('df len:', len(df)))
    if 'SRCID' in df.columns:
        print('>>> {:15} {:.0f}'.format('No of Shots:', df['SRCID'].nunique()))

    if 'GRPID' in df.columns:
        print('>>> {:15} {:.0f}'.format('No of Groups:', df['GRPID'].nunique()))

    info_min = '>>> min'
    info_max = '>>> max'
    info_line = ' ' * 7
    for c in ['SRCX', 'SRCY', 'SRCZ']:
        if c not in df.columns:
            continue
        info_min += " |'{}' {:12.2f}|".format(c[-1], df[c].min())
        info_max += " |'{}' {:12.2f}|".format(c[-1], df[c].max())
        info_line += '-' * 19

    if any(np.intersect1d(['SRCX', 'SRCY', 'SRCZ'], df.columns)):
        print('\n')
        print('>>>>>> Source Geometry Info')
        print(info_min)
        print(info_line)
        print(info_max)

    info_min = '>>> min'
    info_max = '>>> max'
    info_line = ' ' * 7
    for c in ['GRPX', 'GRPY', 'GRPZ']:
        if c not in df.columns:
            continue
        info_min += " |'{}' {:12.2f}|".format(c[-1], df[c].min())
        info_max += " |'{}' {:12.2f}|".format(c[-1], df[c].max())
        info_line += '-' * 19

    if any(np.intersect1d(['GRPX', 'GRPY', 'GRPZ'], df.columns)):
        print('\n')
        print('>>>>>> Group Geometry Info')
        print(info_min)
        print(info_line)
        print(info_max)

    print('>' * 55)


def read_sgy_traces(filename, idx, verbose=True, ignore_geometry=True):
    """
    Reading set of traces by its ID from '.*sgy' file. Reading by 'segyio'.
    :param filename: str path to sgy file
    :param idx: 1D list or array of traces ID
    :param verbose: show reading progress bar True/False
    :param ignore_geometry: ignore geometry checking of 'sgy' True/False
    :return: 2D numpy array of traces (nr - num. of traces, ns - num. of samples)
    """
    data = []
    if verbose:
        iteration = tqdm(idx)
    else:
        iteration = idx

    with segyio.open(filename, ignore_geometry=ignore_geometry) as src:
        for i in iteration:
            tmp = src.trace[i]
            data.append(tmp)

    return np.array(data, ndmin=2, dtype=np.float32)


def read_segy_file_obspy(filename):
    """
    Read segy with obspy
    :param filename:
    :return:
    """
    segy = _read_segy(filename)
    return np.array([x.data for x in segy], ndmin=2, dtype=np.float32)


def read_header_segyio(filename, fields=None, ignore_geometry=True, converter=SEGYIO_HEADER_ITEMS, verbose=False):
    """
    Reading header of 'sgy' with 'segyio'.
    :param filename: str path to sgy file
    :param ignore_geometry: ignore_geometry: ignore geometry checking of 'sgy' True/False
    :param fields: list of 'sgy' headers to use. Default :
            'EnergySourcePoint',
            'SourceX',
            'SourceY',
            'GroupX',
            'GroupY',
            'offset',
            'INLINE_3D',
            'CROSSLINE_3D',
    :param converter: rename column names to be more useful
    :param verbose: ...
    :return: pandas DataFrame
    """
    if not fields:
        fields = list(SEGYIO_HEADER_ITEMS.keys())
    head = {}
    with segyio.open(filename, ignore_geometry=ignore_geometry) as segyfile:
        for h in fields:
            column = converter[h]
            head[column] = segyfile.attributes(eval('segyio.TraceField.{}'.format(h)))[:]
        df = pd.DataFrame(head)

    df['IDX'] = df.index
    return df


def read_header_segyio_full(filename, ignore_geometry=True, drop_nonunique=True):
    """
    Read all fields of segy file by segyio to DataFrame
    :param filename:
    :param ignore_geometry:
    :return:
    """

    with segyio.open(filename, ignore_geometry=ignore_geometry) as segyfile:
        columns = [str(x) for x in segyfile.header[0].keys()]

        values = [dict(x.items()).values() for x in segyfile.header]

    header = pd.DataFrame(values, columns=columns)
    header['IDX'] = header.index

    header = header.T[header.nunique() > 1].T
    header = header.rename(columns=SEGYIO_HEADER_ITEMS)

    return header


def read_seisee_header_info(filename):
    """
    Read header of 'sgy' header written by Seisee
    :param filename:
    :return:
    """
    header_info = []
    i = 0
    with open(filename, "r") as f:
        while True:
            line = f.readline()

            if line.startswith("+-"):
                break

            line = line.replace("*", " ")
            line = line.replace("+", " ")
            line = line[8:]
            line = " ".join(line.split())
            header_info.append([i, line])
            i += 1
    return header_info


def read_header_seisee(filename, fields=None, converter=SEISEE_HEADER_ITEMS, verbose=False):
    """
    Read Seisee header to Pandas DataFrame
    :param filename:
    :param fields:
    :param converter:
    :param verbose:
    :return:
    """
    if not fields:
        fields = list(SEISEE_HEADER_ITEMS.keys())

    header_info = read_seisee_header_info(filename)
    use_cols = [x[0] for x in header_info if x[1] in fields]
    names = [converter[x[1]] for x in header_info if x[1] in fields]
    skip_rows = len(header_info) + 1

    df = pd.read_csv(
        filename,
        skiprows=skip_rows,
        sep="\s+",
        header=None,
        usecols=use_cols,
        names=names,
        dtype=int,
    )
    df["IDX"] -= 1
    return df
