from .header import BIN_HEADER_DESCRIPTOR , TRACE_HEADER_DESCRIPTOR
from .read import read_traces, read_textual_header, read_trace_header, read_bin_header


class Segy:
    def __init__(
            self,
            file_name,
            survey_type='',
            endian='>',
            bin_header_descriptor=BIN_HEADER_DESCRIPTOR,
            trace_header_descriptor=TRACE_HEADER_DESCRIPTOR,
            read_traces_on_init=True,
            verbose=True,
    ):
        self.bin_header_descriptor = bin_header_descriptor
        self.trace_header_descriptor = trace_header_descriptor
        self.survey_type = survey_type
        self.file_name = file_name
        self.textual_header = read_textual_header(file_name)
        self.endian = endian
        self.bin_header = read_bin_header(
            file_name,
            bin_descriptor=bin_header_descriptor,
            endian='>',
        )

        self.traces = None
        self.trace_headers = None

        if read_traces_on_init:
            self.read_traces(file_name, verbose=verbose)

    def read_traces(self, index=None, verbose=True, **kwargs):
        self.traces = read_traces(
            self.file_name,
            bin_header=self.bin_header,
            index=index,
            endian=self.endian,
            verbose=verbose,
            **kwargs,
        )

        self.trace_headers = read_trace_header(
            self.file_name,
            trace_descriptor=self.trace_header_descriptor,
            bin_header=self.bin_header,
            index=index,
            endian=self.endian,
            verbose=verbose,
        )


