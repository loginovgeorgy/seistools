import os
import struct
import numpy as np

from .header import *

# endian='>' # Big Endian  # modified by A Squelch
# endian='<' # Little Endian
# endian='=' # Native


BYTES_FOR_TEXTUAL_HEADER = 3200
BYTES_FOR_SGY_HEADER = 400
BYTES_FOR_TRACE_HEADER = 240
BYTES_FOR_HEADER = BYTES_FOR_TEXTUAL_HEADER + BYTES_FOR_SGY_HEADER


def _read_file(file_name, start=0, n_bytes=-1):
    with open(file_name, 'rb') as f:
        f.seek(start)
        data = f.read(n_bytes)
    return data


def _get_value(
        data,
        index,
        c_type='i',
        endian='>',
        samples=1
):
    if isinstance(c_type, type(None)):
        raise TypeError(
            (
                'The given c-type was ({}).'
                'C-type must be one of the following: {}.'
                'Please see https://docs.python.org/3/library/struct.html'
            ).format(c_type, str(np.unique(list(TRACES_SAMPLES_FORMAT.values()))))
        )
    try:
        size = struct.calcsize(c_type)

    except:
        raise TypeError(
            (
                'The given c-type was ({}).'
                'C-type must be one of the following: {}.'
                'Please see https://docs.python.org/3/library/struct.html'
            ).format(c_type, str(np.unique(list(TRACES_SAMPLES_FORMAT.values()))))
        )

    c_format = "{}{}{}".format(endian, samples, c_type)
    index_end = index + size * samples
    buffer = data[index:index_end]
    value = struct.unpack(c_format, buffer)
    return value


def read_textual_header(file_name):
    data = _read_file(file_name, start=0, n_bytes=BYTES_FOR_TEXTUAL_HEADER)
    return data.decode(encoding='cp500')


def read_bin_header(
        file_name,
        bin_descriptor=None,
        endian='>',
        **kwargs
):
    if isinstance(bin_descriptor, type(None)):
        bin_descriptor = BIN_HEADER_DESCRIPTOR.copy()

    data = _read_file(file_name, BYTES_FOR_TEXTUAL_HEADER, BYTES_FOR_SGY_HEADER)
    bin_header = {}
    for key in bin_descriptor:
        index = bin_descriptor[key]['pos']
        c_type = bin_descriptor[key]['type']

        value = _get_value(data, index, c_type=c_type, endian=endian, samples=1)
        bin_header[key] = value[0]

    return bin_header


def _read_traces(
        file_name,
        bin_header=None,
        index=None,
        endian='>',
        samples_format=None,
        verbose=True,
        method='',
        **kwargs
):
    file_size = os.path.getsize(file_name)

    if not isinstance(bin_header, dict):
        bin_header = read_bin_header(file_name, endian=endian, **kwargs)

    sgy_revision_number = bin_header.get('SgyRevision', None)

    if isinstance(samples_format, type(None)):
        samples_format_code = bin_header.get('DataSampleFormatCode', None)
        samples_format = TRACES_SAMPLES_FORMAT.get(samples_format_code, None)

    no_of_samples = bin_header.get('NumberOfSamples', None)
    no_of_traces = bin_header.get('TracePerRecord', None)

    if isinstance(index, type(None)):
        j_traces = np.arange(no_of_traces)
    else:
        if not isinstance(index, list):
            j_traces = [index]
        else:
            j_traces = index

    check_variables = dict(
        read_traces=len(j_traces),
        no_of_traces=no_of_traces,
        file_name=file_name,
        file_size=file_size,
        sgy_revision_number=sgy_revision_number,
        no_of_samples=no_of_samples,
        samples_format=samples_format
    )

    message = """
    {msg_type}Reading SEG-Y {method} for:
    {read_traces} traces (total: {no_of_traces})
    File: {file_name} of ({file_size}) bytes 
    SEG-Y Rev.: {sgy_revision_number}
    No. of samples: {no_of_samples}
    C-Format: {samples_format}
    """

    if any([isinstance(x, type(None)) for x in check_variables.values()]):
        raise Exception(
            message.format(
                msg_type='Error! ',
                method=method,
                **check_variables,
            )
        )

    bytes_per_sample = struct.calcsize(samples_format)
    bytes_per_trace = bytes_per_sample * no_of_samples

    _no_of_traces = (file_size - BYTES_FOR_HEADER) / (bytes_per_trace + BYTES_FOR_TRACE_HEADER)
    if _no_of_traces != no_of_traces:
        Warning(
            message.format(
                msg_type='File size is incompatible with No. of traces! \n',
                method=method,
                **check_variables,
            )
        )

    if verbose:
        print(
            message.format(
                msg_type='',
                method=method,
                **check_variables,
            )
        )
    check_variables.update(
        dict(
            bytes_per_sample=bytes_per_sample,
            bytes_per_trace=bytes_per_trace,
            j_traces=j_traces,
        )
    )
    return check_variables


def read_trace_header(
        file_name,
        trace_descriptor=None,
        bin_header=None,
        index=None,
        endian='>',
        samples_format=None,
        verbose=False,
        **kwargs
):
    if isinstance(trace_descriptor, type(None)):
        trace_descriptor = TRACE_HEADER_DESCRIPTOR

    check_variables = _read_traces(
        file_name,
        bin_header=bin_header,
        index=index,
        endian=endian,
        samples_format=samples_format,
        verbose=verbose,
        method='Trace Header(s)',
        **kwargs
    )

    j_traces = check_variables['j_traces']
    bytes_per_trace = check_variables['bytes_per_trace']

    trace_header = {}

    for key in trace_descriptor:
        trace_header[key] = []
        pos = trace_descriptor[key]['pos']
        c_type = trace_descriptor[key]['type']
        size = struct.calcsize(c_type)

        for jt in j_traces:
            _pos = pos + BYTES_FOR_HEADER + (bytes_per_trace + BYTES_FOR_TRACE_HEADER) * jt
            data = _read_file(file_name, start=_pos, n_bytes=size)
            value = _get_value(data, 0, c_type=c_type, endian=endian, samples=1)
            trace_header[key].append(value[0])

    return trace_header


def read_traces(
        file_name,
        bin_header=None,
        index=None,
        endian='>',
        samples_format=None,
        verbose=True,
        **kwargs
):
    check_variables = _read_traces(
        file_name,
        bin_header=bin_header,
        index=index,
        endian=endian,
        samples_format=samples_format,
        verbose=verbose,
        method='Trace Data',
        **kwargs
    )

    j_traces = check_variables['j_traces']
    bytes_per_trace = check_variables['bytes_per_trace']
    bytes_per_sample = check_variables['bytes_per_sample']
    no_of_traces = check_variables['no_of_traces']
    no_of_samples = check_variables['no_of_samples']
    samples_format = check_variables['samples_format']

    traces = np.zeros((no_of_traces, no_of_samples))
    for jt in j_traces:
        pos = BYTES_FOR_HEADER + BYTES_FOR_TRACE_HEADER + (bytes_per_trace + BYTES_FOR_TRACE_HEADER) * jt

        data = _read_file(file_name, start=pos, n_bytes=no_of_samples * bytes_per_sample)

        values = _get_value(data, 0, c_type=samples_format, endian=endian, samples=no_of_samples)
        values = np.float32(values)
        traces[jt] = np.squeeze(values)

    return traces
