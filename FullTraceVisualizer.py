# import functions as f
root = "D:\\OneDrive - University of Cambridge\\Usb_stick\\Projects\\Code\\"
import sys

sys.path.append(root)
import misc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# reimport misc
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from nptdms import TdmsFile
import glob
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import RectangleSelector, TextBox, Button


class FTV(FigureCanvas):
    def __init__(self, initialres=1e3, maxres=1e6, min_length=10000):
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        plt.rcParams["svg.fonttype"] = "none"
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        super().__init__(self.fig)
        self.initialres = initialres
        self.maxres = maxres
        self.factor = self.maxres / self.initialres
        self.min_length = min_length
        # if all_data is a string to a folder and ends with .tdms
        self.n_start = 0
        self.n_end = 100

    def load_data_str(self, location, min_length=None, start_index=0, end_index=100):
        if min_length is None:
            min_length = self.min_length
        if isinstance(location, str):
            if location.endswith(".tdms") or location.endswith("/"):
                if location.endswith(".tdms"):
                    files = glob.glob(location)
                elif location.endswith("/"):
                    files = glob.glob(location + "*.tdms")
                self.loadfromstringlist(files, min_length)
                self.init_plot()
                self.files = files
            else:
                print("Please provide a valid path to a folder containing .tdms files")

    def save_data(self, path):
        np.save(path, self.all_data)
        print("Data saved to", path)

    def load_data(self, path):
        self.all_data = np.load(path)
        self.init_plot()

    """
    def detect_outlier_regions(self, threshold=1e-8, window=100):
    """

    def init_plot(self):
        self.ax.clear()
        # Initialize the plot
        # self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.XX = np.linspace(
            0,
            len(self.all_data) / self.maxres,
            int(len(self.all_data) / self.initialres),
        )
        self.lowres_data = self.all_data[:: int(self.initialres)].copy()
        (self.lowresline,) = self.ax.plot(self.XX, self.lowres_data, color="C0")
        self.hireslines = []
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Current (nA)")

        # Connect the function to the event
        self.rectzoom = RectangleSelector(
            self.ax,
            self.on_select_rect,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
        )
        self.fig.canvas.mpl_connect("key_press_event", self.on_click)
        # add the reset view button
        self.ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
        self.breset = plt.Button(self.ax_reset, "Reset view")
        self.breset.on_clicked(self.reset_view)
        # add buttons that skips to the next 100 files
        self.ax_next = plt.axes([0.6, 0.025, 0.1, 0.04])
        self.bnext = plt.Button(self.ax_next, "Next 100 files")
        self.bnext.on_clicked(self.next_files)
        # add textfield for custom n_start
        self.ax_nstart = plt.axes([0.4, 0.025, 0.1, 0.04])
        self.text_nstart = TextBox(self.ax_nstart, "n_start")
        self.text_nstart.on_submit(self.update_nstart)

        self.fig.canvas.draw_idle()
        self.draw()

    def update_nstart(self, event):
        self.n_start = int(event)
        self.n_end = self.n_start + 100
        self.loadfromstringlist(self.files, self.min_length, self.n_start, self.n_end)
        self.init_plot()
        self.fig.canvas.draw_idle()

    def loadfromstringlist(
        self, files, min_length=None, start_index=0, end_index=100, preload=False
    ):
        if min_length is None:
            min_length = self.min_length
        l = end_index - start_index
        data = [[] for i in range(l)]
        # check that not empty
        if l == 0:
            print("No files found in the folder")
            return
        with ThreadPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(
                        lambda f: self.process_file(f, min_length),
                        files[start_index:end_index],
                    ),  # noqa
                    total=l,
                    desc="Loading files",
                    unit="file",
                )
            )

        # Flatten the results
        data = [trace for sublist in results for trace in sublist]

        # Compute lengths and total length
        lengths = [len(trace) for trace in data]
        total_length = sum(lengths)

        # Preallocate numpy array
        all_data = np.zeros(total_length, dtype=float)
        start = 0
        for trace in data:
            all_data[start : start + len(trace)] = trace
            start += len(trace)
        if preload:
            self.preload_data = all_data
        else:
            self.all_data = all_data

    def process_file(self, file, min_length=None):
        if min_length is None:
            min_length = self.min_length
        tdms_file = TdmsFile.read(file)
        data = []
        for group in tdms_file.groups():
            for channel in group.channels():
                if len(channel) > min_length:
                    trace = channel[:]
                    data.append(trace)
        del tdms_file
        return data

    def next_files(self, event):
        self.n_start += 100
        self.n_end += 100
        # self.all_data = self.preload_data
        self.loadfromstringlist(self.files, self.min_length, self.n_start, self.n_end)
        self.init_plot()
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.key == "f":
            self.plot_high_res()
        elif event.key == "r":
            self.reset_view()
        elif event.key == "l":
            misc.zoom_in_x(self.ax)
        elif event.key == "j":
            misc.zoom_out_x(self.ax)
        elif event.key == "a":
            misc.pan_left(self.ax)
        elif event.key == "d":
            misc.pan_right(self.ax)
        elif event.key == "i":
            misc.zoom_in_y(self.ax)
        elif event.key == "k":
            misc.zoom_out_y(self.ax)
        elif event.key == "w":
            misc.pan_up(self.ax)
        elif event.key == "s":
            misc.pan_down(self.ax)
        elif event.key == "u":
            misc.zoom_out_both(self.ax)
        elif event.key == "o":
            misc.zoom_in_both(self.ax)
        elif event.key == "h":
            misc.zoom_out_x_10times(self.ax)
        elif event.key == "n":
            self.calculate_noise()
        # draw
        self.fig.canvas.draw_idle()

    def on_select_rect(self, eclick, erelease):
        # Zoom into the selected rectangle
        # but check that the rectangle is not too small
        # get current axis limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xsize = np.abs(xlim[1] - xlim[0])
        ysize = np.abs(ylim[1] - ylim[0])
        if (
            np.abs(eclick.x - erelease.x) > 0.01 * xsize
            and np.abs(eclick.y - erelease.y) > 0.01 * ysize
        ):
            self.set_axes(
                None, [eclick.xdata, erelease.xdata], [eclick.ydata, erelease.ydata]
            )

    def set_axes(self, event=None, cust_xlim=None, cust_ylim=None, paddingfactor=0.2):
        if cust_xlim is None:
            minx, maxx = np.min(self.dataset_lowdim1), np.max(self.dataset_lowdim1)
        else:
            minx, maxx = cust_xlim
        if cust_ylim is None:
            miny, maxy = np.min(self.dataset_lowdim2), np.max(self.dataset_lowdim2)
        else:
            miny, maxy = cust_ylim
        if minx == maxx:
            diffx = 0.1
        else:
            diffx = maxx - minx
        if miny == maxy:
            diffy = 0.1
        else:
            diffy = maxy - miny
        if cust_xlim is not None:
            paddingfactor = 0
        self.ax.set_xlim([minx - paddingfactor * diffx, maxx + paddingfactor * diffx])
        self.ax.set_ylim([miny - paddingfactor * diffy, maxy + paddingfactor * diffy])
        self.fig.canvas.draw_idle()

    def plot_high_res(self):
        xlim = self.ax.get_xlim()
        X0, X1 = xlim
        diff = X1 - X0
        visible_points = (X1 - X0) * self.initialres

        X = np.linspace(X0 - diff, X1 + diff, int(visible_points * self.factor) * 3)
        Xindices = range(int((X0 - diff) * self.maxres), int((X1 + diff) * self.maxres))

        shapex = X.shape
        shapey = self.all_data[Xindices].shape
        if shapex[0] != shapey[0]:
            diff = shapex[0] - shapey[0]
            if diff > 0:
                X = X[:-diff]
            elif diff < 0:
                for _ in range(abs(diff)):
                    X = np.append(X, X[-1] + 1)

        (hiresline,) = self.ax.plot(X, self.all_data[Xindices], color="C0")
        # delete old hireslines that are close to the new hiresline
        for line in self.hireslines:
            if np.abs(line.get_xdata()[0] - hiresline.get_xdata()[0]) < 1e-6:
                line.remove()
                self.hireslines.remove(line)
        self.hireslines.append(hiresline)

        minindex = int((X0 - diff) * self.initialres)
        maxindex = int((X1 - diff) * self.initialres)
        self.lowres_data[minindex:maxindex] = np.nan
        self.lowresline.set_ydata(self.lowres_data)
        print("Hi-res plot added")
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()

    def plot_low_res(self):
        self.lowresline.set_ydata(self.all_data[:: int(self.initialres)])
        for line in self.hireslines:
            line.remove()
        self.hireslines.clear()

    def reset_view(self, event=None):
        # Disconnect and remove the previous RectangleSelector if it exists
        if self.rectzoom is not None:
            self.rectzoom.set_active(False)
            self.rectzoom = None
        ymax = np.max(self.all_data)
        ymin = np.min(self.all_data)
        diff = ymax - ymin
        xlen = len(self.all_data) / self.maxres
        self.ax.set_xlim([-0.05 * xlen, 1.05 * xlen])
        self.ax.set_ylim([ymin - 0.1 * diff, ymax + 0.1 * diff])
        self.fig.canvas.draw_idle()

        self.rectzoom = RectangleSelector(
            self.ax,
            self.on_select_rect,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
        )

    def calculate_noise(self):
        # Calculate the noise of the signal
        # get indices of current view
        xlim = self.ax.get_xlim()
        X0, X1 = xlim
        X0int = int(X0 * self.maxres)
        X1int = int(X1 * self.maxres)
        # get the data
        data = self.all_data[X0int:X1int]
        noise = np.std(data)
        print(
            f"The noise of the signal in the current viewing window is {noise:.5f} nA"
        )
