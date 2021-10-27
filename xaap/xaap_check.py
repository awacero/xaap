


import os, sys
from pathlib import Path
import pyqtgraph as pg

from pyqtgraph.Qt import QtGui, QtCore

from pyqtgraph.Qt.QtGui import QTableWidget



import logging, logging.config

from pyqtgraph.parametertree.ParameterTree import ParameterTree

xaap_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])))
xaap_config_dir = Path("%s/%s" %(xaap_dir,"config"))
logging.config.fileConfig(xaap_config_dir / "logging.ini" ,disable_existing_loggers=False)
logger = logging.getLogger('stdout')
logger.setLevel(logging.INFO)


class xaap_check(QtGui.QWidget):


    def __init__(self):

        logger.info("Continuation of all this #$%&")

        QtGui.QWidget.__init__(self)

        self.setupGUI()


    def setupGUI(self):

        self.layout = QtGui.QVBoxLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.layout)


        self.tree =  ParameterTree()
        self.tree2 = ParameterTree()
        self.datetime_axis_1 = pg.graphicsItems.DateAxisItem.DateAxisItem(orientation = 'bottom')
        self.main_plot = pg.GraphicsLayoutWidget()
        self.side_plot = pg.GraphicsLayoutWidget()
        self.time_pw = self.main_plot.addPlot(row=1, col=0,axisItems={'bottom': self.datetime_axis_1})

        self.table_widget = QTableWidget()
        self.table_widget.setRowCount(4)
        self.table_widget.setColumnCount(4)


        '''horizontal splitter splits windows in 2 verticall'''
        self.splitter_horizontal = QtGui.QSplitter()
        self.splitter_horizontal.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.layout.addWidget(self.splitter_horizontal)
        
        self.splitter_horizontal.addWidget(self.main_plot)
        self.splitter_horizontal.addWidget(self.side_plot) 

        self.splitter_vertical = QtGui.QSplitter()
        self.splitter_vertical.setOrientation(QtCore.Qt.Orientation.Vertical)

        #self.splitter_vertical.addWidget(self.splitter_horizontal)

        #self.splitter_vertical.addWidget(self.tree2)
        self.splitter_vertical.addWidget(self.table_widget)

        self.layout.addWidget(self.splitter_vertical)

        '''
        self.splitter_vertical = QtGui.QSplitter()
        self.splitter_vertical.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.layout.addWidget(self.splitter_vertical)
        self.splitter_vertical.addWidget(self.tree)
        self.splitter_vertical.addWidget(self.plot_window)
        '''






if __name__ == '__main__':
    app = pg.mkQApp()
    app.setStyleSheet("""
    QWidget {font-size: 15px}
    QMenu {font-size: 10px}
    QMenu QWidget {font-size: 10px}
""")

    win = xaap_check()
    win.setWindowTitle("xaap_check")
    win.show()
    win.resize(1100,700)

    pg.exec()