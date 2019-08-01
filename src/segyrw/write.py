import struct
import numpy as np
import sys
from .header import *


def _write_file(file_name, data, start = 0):
    with open(file_name, 'rb+') as f:
        f.seek(start, 0)
        f.write(data)
        f.close()
    return


def write_textual_header(file_name, text):
    data = text.encode(encoding='cp500')
    if sys.getsizeof(data) > BYTES_FOR_TEXTUAL_HEADER+34:
        raise ValueError(
            (
                'The size of the required size: {}.'
            ).format(BYTES_FOR_TEXTUAL_HEADER)
        )
    with open(file_name, 'wb') as f:
        f.truncate()
        f.seek(0)
        f.write(data)
        f.close()
    return


def _add_value(data, c_type='i', endian='>', samples=1):
    if isinstance(c_type, type(None)):
        raise TypeError(
            (
                'The given c-type was ({}).'
                'C-type must be one of the following: {}.'
                'Please see https://docs.python.org/3/library/struct.html'
            ).format(c_type, str(np.unique(list(TRACES_SAMPLES_FORMAT.values()))))
        )

    c_format = "{}{}{}".format(endian, samples, c_type)

    if type(data) == np.ndarray:
        value = struct.pack(c_format, *data)
    else:
        value = struct.pack(c_format, data)

    return value


def check_size(all_pos, types, required_size):
    max_pos = np.max(all_pos)
    arg_max = np.argmax(all_pos)
    size_type = struct.calcsize(types[arg_max])
    if max_pos + size_type > required_size:
        raise ValueError(
            (
                'The size of the required size: {}.'
            ).format(required_size)
        )
    return


def write_bin_header(file_name, data, bin_descriptor=None, endian='>'):
    if isinstance(bin_descriptor, type(None)):
        bin_descriptor = BIN_HEADER_DESCRIPTOR.copy()
    all_pos = []
    types = []
    for i, key in enumerate(bin_descriptor):
        if key not in data:
            continue
        index = bin_descriptor[key]['pos']
        c_type = bin_descriptor[key]['type']
        bin_head = _add_value(data[key], c_type, endian=endian, samples=1)
        _write_file(file_name, bin_head, start=BYTES_FOR_TEXTUAL_HEADER + index)

        all_pos.append(index)
        types.append(c_type)
    check_size(all_pos, types, BYTES_FOR_SGY_HEADER)

    return


def _write_trace(file_name, bin_header, samples_format, method):
    no_of_samples = bin_header.get('NumberOfSamples', None)
    no_of_traces = bin_header.get('TracePerRecord', None)
    if isinstance(samples_format, type(None)):
        samples_format_code = bin_header.get('DataSampleFormatCode')
        samples_format = TRACES_SAMPLES_FORMAT.get(samples_format_code, None)
    bytes_per_sample = struct.calcsize(samples_format)
    bytes_per_trace = bytes_per_sample * no_of_samples

    message = """
        {msg_type}Reading SEG-Y {method} for:
        {read_traces} traces (total: {no_of_traces})
        File: {file_name} of ({file_size}) bytes 
        SEG-Y Rev.: {sgy_revision_number}
        No. of samples: {no_of_samples}
        C-Format: {samples_format}
        """

    check_variables = dict(
        no_of_samples = no_of_samples,
        no_of_traces = no_of_traces,
        file_name = file_name,
        samples_format = samples_format,
        bytes_per_sample = bytes_per_sample,
        bytes_per_trace = bytes_per_trace,
    )
    if any([isinstance(x, type(None)) for x in check_variables.values()]):
        raise Exception(
            message.format(
                msg_type='Error! ',
                method=method,
                **check_variables,
            )
        )
    return check_variables


def write_trace_header_traces(file_name, data, traces, trace_descriptor = None, bin_header = None, endian = '>', samples_format=None, **kwargs):
    if isinstance(trace_descriptor, type(None)):
        trace_descriptor = TRACE_HEADER_DESCRIPTOR

    check_variable = _write_trace(file_name, bin_header, samples_format, method='trace_header')
    bytes_per_trace = check_variable['bytes_per_trace']
    no_of_traces = check_variable['no_of_traces']
    trace_header = {}

    all_pos = []
    types = []
    for j in range(no_of_traces):
        for key in trace_descriptor:
            if key not in data:
                continue
            pos = trace_descriptor[key]['pos']
            c_type = trace_descriptor[key]['type']

            _pos = pos + BYTES_FOR_HEADER + (bytes_per_trace + BYTES_FOR_TRACE_HEADER)*j
            value = _add_value(data[key][j], c_type, endian, samples=1)
            trace_header[key] = value
            _write_file(file_name, trace_header[key], start=_pos)

            if j == 0:
                all_pos.append(pos)
                types.append(c_type)
                check_size(all_pos, types, BYTES_FOR_TRACE_HEADER)

        write_traces(file_name, traces, bin_header, j)
    return


def write_traces(file_name, data, bin_header, index, endian = '>', samples_format=None, method = 'Trace_Data', **kwargs):

    data = np.array(data)
    check_variable = _write_trace(file_name, bin_header, samples_format, method)
    bytes_per_trace = check_variable['bytes_per_trace']
    samples_format = check_variable['samples_format']
    no_of_samples = check_variable['no_of_samples']

    pos = BYTES_FOR_HEADER + BYTES_FOR_TRACE_HEADER + (bytes_per_trace + BYTES_FOR_TRACE_HEADER)*index
    data[index] = np.squeeze(data[index])

    value = _add_value(data[index], c_type=samples_format, endian=endian, samples=no_of_samples)
    _write_file(file_name, value, start=pos)
    return


def write_sgy(file_name, text, bin_head, trace_header, traces):
    write_textual_header(file_name, text)
    write_bin_header(file_name, bin_head)
    write_trace_header_traces(file_name, trace_header, traces, bin_header=bin_head)

