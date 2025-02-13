# main.py
import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QTabWidget,
    QFileDialog,
    QPushButton,
    QMessageBox,
)
from PyQt6 import QtCore
from FullTraceVisualizer import FTV


class App(QMainWindow):
    """Main application class for the Interactive Plot GUI."""

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        """Initializes the UI components."""
        self.setWindowTitle("Interactive Plot GUI")
        self.setGeometry(50, 50, 800, 600)

        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)

        layout = QVBoxLayout(main_widget)
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        self.tab1 = QWidget()

        self.tab_widget.addTab(self.tab1, "Full Trace Visualizer")

        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data)
        button_layout.addWidget(self.load_button)

        self.tab1_layout = QVBoxLayout(self.tab1)
        self.FTVinstance = FTV()
        self.FTVinstance.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.FTVinstance.setFocus()
        self.tab1_layout.addWidget(self.FTVinstance)
        self.current_tab = 0

    def on_tab_changed(self, index):
        if index == self.tab_widget.indexOf(self.tab1):
            self.FTVinstance.setFocus()

    def load_data(self, event):
        """Selects folder containing TDMS files."""
        file_dialog = QFileDialog()
        folder = file_dialog.getExistingDirectory(None, "Open TDMS Folder")
        if folder:
            print("Selected path:" + folder)
            self.FTVinstance.load_data_str(folder + "/")

    def save_dict(self, event):
        """Saves the dictionary to a file."""
        return

    '''
    def show_warning(self, message):
        """Displays a warning message."""
        warning_box = QMessageBox()
        warning_box.setIcon(QMessageBox.Icon.Warning)
        warning_box.setWindowTitle("Warning")
        warning_box.setText(message)
        warning_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        warning_box.exec()
    '''


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    ex.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
    ex.setFocus()
    ex.show()
    sys.exit(app.exec())
