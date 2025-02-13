import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLineEdit,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QCheckBox,
)
from PyQt6 import QtCore
from matplotlib.widgets import Button, Rectangle


# Class to create and manage a progress bar
class ProgressBar:
    def __init__(self, ax, max_val):
        self.ax = ax
        self.max_val = max_val
        self.progress = 0
        self.bar = Rectangle((0, 0), 0, 1, color="blue")
        self.ax.add_patch(self.bar)
        self.ax.set_xlim(0, max_val)
        self.ax.set_ylim(0, 1)
        self.ax.axis("off")

    def update(self, progress):
        self.progress = progress
        self.bar.set_width(progress)
        self.ax.figure.canvas.draw()


# Define root directory for saving figures
root = "D:/OneDrive - Cambridge/OneDrive - University of Cambridge/Usb_stick/Code"


# Function to save figures in multiple formats
def save_fig(name, fig_object=None, folder="figures"):
    """
    Saves svg, png and reaccessible
    """
    dir = root + "/" + folder + "/"
    Path(dir).mkdir(parents=True, exist_ok=True)
    if fig_object is None:
        print(
            "No figure object given, please put plt.figure before where you started figure or provide fig"
        )
    else:
        fig_object.savefig(root + "/" + folder + "/" + name + ".png", dpi=300)
        fig_object.savefig(root + "/" + folder + "/" + name + ".svg")
        fig_object.savefig(root + "/" + folder + "/" + name + ".pdf")
        pickle.dump(
            fig_object, open(root + "/" + folder + "/" + name + ".pickle", "wb")
        )


# Function to load saved figures
def load_fig(name, folder="figures"):
    """
    Loads a figure from the figures folder
    """
    fig_object = pickle.load(open(root + "/" + folder + "/" + name, "wb"))
    return fig_object


# Function to handle text input submission
def on_submit(text, additional_param):
    print(f"Text entered: {text}")
    print(f"Additional parameter: {additional_param}")


# Function to calculate the width of an event in a trace
def get_width(trace, wid=0.1):
    """
    Function that returns the width of the event in the trace
    Does so by convolution with step function and taking maxima

    Parameters
    ----------
    trace : array_like
        Signal as a 1D array of values.
    wid : float, optional
        Width of the step function as a fraction of the trace length. The default is 0.25.
    """
    # Obtain step function and padding length from absolute length
    tlen = len(trace)
    stepwid = int(tlen * wid)
    # 1.
    step = np.concatenate((np.ones(stepwid) / stepwid, -np.ones(stepwid) / stepwid))
    conv = np.convolve(trace, step, mode="same")
    # 2.
    minconv = np.argmin(conv)
    maxconv = np.argmax(conv)
    # 3.
    return minconv, maxconv


# Function to add a scale bar to a plot
def get_plot_scale_bar(
    trace,
    ax,
    points_per_ms=1000,
    cut_steps=[0.05, 0.1, 0.2, 0.5, 1, 2, 5],
    Y0_float=0.9,
    X0_float=0,
    xlabel_offset=0.01,
    ylabel_offset=0.01,
    xlims=None,
    ylims=None,
    manual_scale=None,
    return_annotations=False,
    customx=None,
):
    """
    Adds a scale bar to a plot.

    :param trace: Array of data points for the plot.
    :param ax: The matplotlib axis object to add the scale bar to.
    :param points_per_ms: Number of data points per millisecond.
    :param cut_steps: Thresholds for deciding the scale bar width.
    :param label_steps: Labels corresponding to the cut_steps.
    :param Y0_float, X0_float: Relative positions for the scale bar.
    :param label_offset: Offset for the scale bar label.
    :param xlims, ylims: Limits for the x and y axes.
    :param manual_scale: Tuple (X_scale, Y_scale) to set scale bar manually.
    """

    # Determine the minimum y-value for scaling.
    minim = min(trace) if ylims is None else min(np.min(trace), *ylims)

    # Set vertical size of the scale bar, based on data range.
    cutindex = 0
    while np.abs(minim) > cut_steps[cutindex]:
        cutindex += 1
    Ywid = cut_steps[cutindex] * 0.1
    # Ywid = np.max([int(abs(minim) * 2), 0.5]) * 0.1
    Ylabel = "{:10.2f} nA".format(Ywid)
    Y0 = Y0_float * minim

    # Set initial horizontal position of the scale bar.
    if customx is not None:
        X0 = X0_float * len(trace) + customx[0]
    else:
        X0 = X0_float * len(trace)

    # Set the scale bar width automatically or use manual setting.
    if manual_scale:
        Xwid, Ywid = manual_scale
        Xlabel = f"{Xwid} ms"
        Ylabel = f"{Ywid} nA"
    else:
        for n, c in enumerate(cut_steps):
            if len(trace) < points_per_ms * c:
                Xwid = points_per_ms * c * 0.1
                Xlabel = str(c * 0.1) + " ms"
                break
    Xlabel = "{:10.2f} ms".format(Xwid * 0.001)
    # Set plot limits if specified.
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    # Define points for the scale bar.
    X = [X0, X0, X0 + Xwid]
    Y = [Y0 + Ywid, Y0, Y0]

    # Add scale bar to the plot.
    ax.plot(X, Y, color="black")

    # Annotate the scale bar with labels.
    text_x = ax.annotate(
        Xlabel,
        xy=(X0 + Xwid / 2, Y0 + ylabel_offset * minim),
        color="black",
        fontsize=12,
        ha="center",
        va="top",
    )
    text_y = ax.annotate(
        Ylabel,
        xy=(X0 - xlabel_offset * len(trace), Y0 + Ywid / 2),
        color="black",
        fontsize=12,
        ha="right",
        va="center",
    )
    if return_annotations:
        return text_x, text_y


# Function to increase the size of a checkbox
def increase_checkboxsize(box):
    new_width = 0.1
    new_height = 0.3
    # Adjust the coordinates and size of the cross
    for rect, lines in zip(box.rectangles, box.lines):
        rect.set_width(new_width)
        rect.set_height(new_height)
        x0, y0 = rect.get_xy()
        width, height = new_width, new_height

        # Adjust the lines to fit the rectangle
        x_center = x0 + width / 2
        y_center = y0 + height / 2
        offset_x = width / 3
        offset_y = height / 3

        lines[0].set_data(
            [x_center - offset_x, x_center + offset_x],
            [y_center - offset_y, y_center + offset_y],
        )
        lines[1].set_data(
            [x_center - offset_x, x_center + offset_x],
            [y_center + offset_y, y_center - offset_y],
        )


# Functions to zoom in/out and pan on a plot along x-axis
def zoom_in_x(ax):
    xlim = ax.get_xlim()
    X0, X1 = xlim
    diff = X1 - X0
    ax.set_xlim(X0 + diff * 0.2, X1 - diff * 0.2)


def zoom_out_x(ax):
    xlim = ax.get_xlim()
    X0, X1 = xlim
    diff = X1 - X0
    ax.set_xlim(X0 - diff * 0.2, X1 + diff * 0.2)


def zoom_out_x_10times(ax):
    xlim = ax.get_xlim()
    X0, X1 = xlim
    diff = X1 - X0
    ax.set_xlim(X0 - diff * 10, X1 + diff * 10)


def pan_left(ax):
    xlim = ax.get_xlim()
    X0, X1 = xlim
    diff = X1 - X0
    ax.set_xlim(X0 - diff * 0.2, X1 - diff * 0.2)


def pan_right(ax):
    xlim = ax.get_xlim()
    X0, X1 = xlim
    diff = X1 - X0
    ax.set_xlim(X0 + diff * 0.2, X1 + diff * 0.2)


# Functions to zoom in/out and pan on a plot along y-axis
def zoom_in_y(ax):
    ylim = ax.get_ylim()
    Y0, Y1 = ylim
    diff = Y1 - Y0
    ax.set_ylim(Y0 + diff * 0.2, Y1 - diff * 0.2)


def zoom_out_y(ax):
    ylim = ax.get_ylim()
    Y0, Y1 = ylim
    diff = Y1 - Y0
    ax.set_ylim(Y0 - diff * 0.2, Y1 + diff * 0.2)


def pan_up(ax):
    ylim = ax.get_ylim()
    Y0, Y1 = ylim
    diff = Y1 - Y0
    ax.set_ylim(Y0 + diff * 0.2, Y1 + diff * 0.2)


def pan_down(ax):
    ylim = ax.get_ylim()
    Y0, Y1 = ylim
    diff = Y1 - Y0
    ax.set_ylim(Y0 - diff * 0.2, Y1 - diff * 0.2)


def zoom_out_both(ax):
    zoom_out_x(ax)
    zoom_out_y(ax)


def zoom_in_both(ax):
    zoom_in_x(ax)
    zoom_in_y(ax)


# Dialog class to enter a category name
class CategoryDialog(QDialog):
    category_submitted = QtCore.pyqtSignal(
        str
    )  # Signal emitted when category is submitted

    def __init__(self):
        super().__init__()
        self.initUI()
        self.category_name = ""

    def initUI(self):
        self.setWindowTitle("Enter Category Name")

        self.layout = QVBoxLayout(self)

        self.input_layout = QHBoxLayout()
        self.category_input = QLineEdit(self)
        self.category_input.textChanged.connect(self.update_category_name)
        self.input_layout.addWidget(QLabel("Enter Category:"))
        self.input_layout.addWidget(self.category_input)

        self.layout.addLayout(self.input_layout)

        self.submit_button = QPushButton("Submit", self)
        self.submit_button.clicked.connect(self.accept)
        self.layout.addWidget(self.submit_button)

        self.setLayout(self.layout)

    def update_category_name(self, text):
        self.category_name = text

    def accept(self):
        if self.category_name:
            self.category_submitted.emit(self.category_name)
        super().accept()


# Dialog class to enter a name
class NameDialog(QDialog):
    category_submitted = QtCore.pyqtSignal(
        str
    )  # Signal emitted when category is submitted

    def __init__(self, title=None, label=None, init_text=""):
        super().__init__()
        self.category_name = ""
        self.title = title
        self.label = label
        self.init_text = init_text
        self.initUI()

    def initUI(self):
        if self.title is not None:
            self.setWindowTitle(self.title)
        else:
            self.setWindowTitle("Enter Name")

        self.layout = QVBoxLayout(self)

        self.input_layout = QHBoxLayout()
        self.category_input = QLineEdit(self, text=self.init_text)
        self.category_input.textChanged.connect(self.update_category_name)
        if self.label is not None:
            self.input_layout.addWidget(QLabel(self.label))
        self.input_layout.addWidget(self.category_input)
        self.layout.addLayout(self.input_layout)

        self.submit_button = QPushButton("Submit", self)
        self.submit_button.clicked.connect(self.accept)
        self.layout.addWidget(self.submit_button)

        self.setLayout(self.layout)

    def update_category_name(self, text):
        self.category_name = text

    def accept(self):
        if self.category_name:
            self.category_submitted.emit(self.category_name)
        super().accept()


# Toggle button class for switching between two states
class ToggleButton:
    def __init__(
        self, ax, label_on="Zoom", label_off="Lasso", on_color="red", off_color="green"
    ):
        self.ax = ax
        self.button = Button(ax, label_off)
        self.on_color = on_color
        self.off_color = off_color
        self.zoomactive = False
        self.update_color()
        self.button.on_clicked(self.toggle)
        self.label_on = label_on
        self.label_off = label_off

    def update_color(self):
        color = self.on_color if self.zoomactive else self.off_color
        self.ax.set_facecolor(color)
        self.ax.figure.canvas.draw()

    def toggle(self, event=None):
        self.zoomactive = not self.zoomactive
        self.button.label.set_text(self.label_on if self.zoomactive else self.label_off)
        self.update_color()
