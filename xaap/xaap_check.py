


import os, sys
import pandas as pd
from pathlib import Path
from PyQt5.QtGui import QStandardItemModel, QStandardItem

from PyQt5.QtWidgets import QTableView

import pyqtgraph as pg

from pyqtgraph.Qt import QtGui, QtCore

from pyqtgraph.Qt.QtGui import QTableWidget

from pyqtgraph import TableWidget

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
        self.predictions_file = Path(xaap_dir,"data/classifications/") / "out_xaap_2021.10.25.22.24.53.txt"

        self.setup_model_view()

        self.setupGUI()


    def load_csv_file(self):


        try:
            prediction_data  = pd.read_csv(self.predictions_file,sep=',')
            self.table_widget.setData(prediction_data.to_numpy())
        except Exception as e:
            raise Exception("Error reading prediction file : %s" %str(e))
        
            

        '''
        open_csv = open(self.predictions_file,"r")
        reader = csv.reader(open_csv)
        for i, row in enumerate(csv.reader(open_csv)):
            items = [QStandardItem(item) for item in row]
            self.model.insertRow(i,items)
        '''

    def setup_model_view(self):

        """
        Set up standard model and table view. 
        """
        """
        self.table_widget = QTableWidget()
        self.model = QStandardItemModel()      
        self.table_view = QTableView()
        # For QAbstractItemView.ExtendedSelection = 3
        self.table_view.SelectionMode(3) 
        self.table_view.setModel(self.model)
        """ 

        # Set initial row and column values
        #self.model.setRowCount(3)
        #self.model.setColumnCount(4)
        self.table_widget = TableWidget(editable=True)
        self.load_csv_file()

        """
        self.model = QStandardItemModel()
        self.table_widget.setModel(self.model)
        self.load_csv_file()
        """


    def setupGUI(self):

        self.layout = QtGui.QVBoxLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.layout)


        self.tree =  ParameterTree()
        self.tree2 = ParameterTree()
        self.datetime_axis_1 = pg.graphicsItems.DateAxisItem.DateAxisItem(orientation = 'bottom')
        self.main_plot = pg.GraphicsLayoutWidget()
        self.side_plot = pg.GraphicsLayoutWidget()
        
        self.plot_a = self.side_plot.addPlot(row=0,col=0)
        self.plot_b = self.side_plot.addPlot(row=1,col=0)
        self.plot_c = self.side_plot.addPlot(row=2,col=0)

        self.time_pw = self.main_plot.addPlot(row=1, col=0,axisItems={'bottom': self.datetime_axis_1})

        '''horizontal splitter sides widgets horizontally'''
        self.splitter_horizontal = QtGui.QSplitter()
        self.splitter_horizontal.setOrientation(QtCore.Qt.Orientation.Horizontal)
        '''vertical splitter stacks widgets vertically'''
        self.splitter_vertical = QtGui.QSplitter()
        self.splitter_vertical.setOrientation(QtCore.Qt.Orientation.Vertical)
        
        self.splitter_vertical.addWidget(self.main_plot)
        #self.splitter_vertical.addWidget(self.table_view)
        self.splitter_vertical.addWidget(self.table_widget)

        self.splitter_horizontal.addWidget(self.splitter_vertical)
        self.splitter_horizontal.addWidget(self.side_plot)
        
        self.splitter_horizontal.setSizes([600,200])
        self.splitter_vertical.setSizes([600,200])
        self.layout.addWidget(self.splitter_horizontal)
    


        

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