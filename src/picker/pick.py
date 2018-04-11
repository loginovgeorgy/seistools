import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib import gridspec


class Picker(object):
    def __init__(self, filename_data, filename_picks, auto_save=False):

        self.auto_save = auto_save
        self.filename_data = filename_data
        self.filename_picks = filename_picks
        self.ts = 16000
        self.te = 20000

        self.traces, _ = self.load_data()
        self.picks = self.load_picks()
        self._process_traces()

        self.ns = self.traces.shape[1]
        self.nr = self.traces.shape[0]

        fig, axs = plt.subplots()
        self.axes = axs
        self.fig = fig
        self.plot_traces()
        self.line = self.plot_peaks()
        self.cid = self.axes.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.button == 1: return
        if event.inaxes != self.line.axes: return

        jt = np.int32(np.round(event.xdata))
        jp = np.int32(event.ydata)
        self.picks[jt] = jp
        self.line.set_ydata(self.picks[self.ts:self.te])
        self.line.figure.canvas.draw()
        # self.save_picks()
        if self.auto_save:
            self.save_data()

    def _process_traces(self):
        interval = np.arange(self.ts, self.te)
        self.traces /= np.abs(self.traces[interval]).max()

    def plot_peaks(self):
        line = self.axes.plot(np.arange(self.ts, self.te), self.picks[self.ts:self.te], 'r')[0]
        return line

    def plot_traces(self):
        _tr = np.flipud(self.traces[self.ts: self.te].T)
        _tr /= np.abs(_tr).max(axis=1, keepdims=True)
        plt.imshow(_tr, extent=(self.ts, self.te, 1 , self.ns))
        # for i, j in enumerate(range(self.ts, self.te)):
        #     self.axes.plot(2*self.traces[j] + j, range(self.ns), 'k')

    def load_data(self):

        with open(self.filename_data, 'rb') as handle:
            data = pickle.load(handle)
        traces = data['data']
        picks = data['y']
        return traces, picks

    def load_picks(self):
        # with open(self.filename_picks, 'rb') as handle:
        #     data = pickle.load(handle)
        # picks = data['y']

        df = pd.read_csv(self.filename_picks)
        picks = df['FirstBreak'].values

        return picks

    def save_picks(self):
        data = {'y': self.picks}
        with open(self.filename_picks, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_data(self):
        data = {'data': self.traces, 'y': self.picks}
        with open(self.filename_data, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
