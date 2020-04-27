import os

from PyQt5 import QtGui, QtWidgets, QtCore
from demos.twip_widget import TWIPWidget
from demos.plot_widget import RollingPlotWidget

from twip.model.robot import load_robot_json


class MainWindow(QtWidgets.QMainWindow):
    ''' Realtime TWIP viewer program
    '''

    def __init__(self):
        super(MainWindow, self).__init__()

        # Create TWIP model
        self.twip = load_robot_json( os.path.dirname(os.path.realpath(__file__)) + "/../docs/robot_generic.json")
        self.twip_widget = TWIPWidget(self, self.twip)

        self.resize(1900, 1000)
        # Add layout to put twip_widget in
        wid = QtWidgets.QWidget(self)
        self.setCentralWidget(wid)

        # create layouts
        self.view_layout = QtGui.QHBoxLayout()
        self.twip_layout = QtGui.QHBoxLayout()
        self.plot_layout = QtGui.QVBoxLayout()

        wid.setLayout(self.view_layout)
        self.view_layout.addLayout(self.twip_layout, 2)
        self.view_layout.addLayout(self.plot_layout, 1)

        self.twip_layout.addWidget(self.twip_widget)

        self.tilt_widget = RollingPlotWidget(1, 300)
        self.yaw_widget = RollingPlotWidget(1, 300)
        self.motor_plot_widget = RollingPlotWidget(2, 300)

        self.tilt_widget.set_pen(0, 'r')
        self.tilt_widget.setLabel('left', 'Tilt', units='degrees')
        self.tilt_widget.setLabel('bottom', 'Sample Number')
        self.tilt_widget.showGrid(True, True, 0.5)

        self.yaw_widget.set_pen(0, 'c')
        self.yaw_widget.setLabel('left', 'Yaw', units='degrees')
        self.yaw_widget.setLabel('bottom', 'Sample Number')
        self.yaw_widget.showGrid(True, True, 0.5)

        self.motor_plot_widget.set_pen(0, 'g')
        self.motor_plot_widget.set_pen(1, 'w')
        self.motor_plot_widget.setLabel('left', 'Motor Torque', units='N m')
        self.motor_plot_widget.setLabel('bottom', 'Sample Number')
        self.motor_plot_widget.showGrid(True, True, 0.5)

        self.plot_layout.addWidget(self.tilt_widget)
        self.plot_layout.addWidget(self.yaw_widget)
        self.plot_layout.addWidget(self.motor_plot_widget)

        # wid.setLayout(mainLayout)

        # Setup twip initial state
        dt = 1 / 30
        self.twip.set_IC([0, 0, 0.4, -0.1, 0, 0])
        self.twip.update_current_state(dt, [1 / dt * 0.5, 1 / dt * 0.4, 0, 0])
        self.dt = dt

    def update_twip(self):
        ''' program mainloop method
        '''
        self.twip.update_current_state(self.dt, [0, 0, 0, 0])

        m_l = self.twip.motor_l.get_position_coordinates()[0]
        m_r = self.twip.motor_r.get_position_coordinates()[0]
        self.motor_plot_widget.push_data([m_l, m_r])

        coords = self.twip.get_position_coordinates()
        y = coords[2] * 180 / 3.1415
        t = coords[5] * 180 / 3.1415
        self.tilt_widget.push_data([t])
        self.yaw_widget.push_data([y])

    def update_plot(self):
        # pass
        self.motor_plot_widget.update_plot()
        self.tilt_widget.update_plot()
        self.yaw_widget.update_plot()
        self.twip_widget.draw_twip()


app = QtWidgets.QApplication(['TWIP Viewer'])
window = MainWindow()

sim_timer = QtCore.QTimer()
sim_timer.timeout.connect(window.update_twip)
sim_timer.start(1 / 60 * 1000)

plot_timer = QtCore.QTimer()
plot_timer.timeout.connect(window.update_plot)
plot_timer.start(1 / 40 * 1000)

window.show()

app.exec_()
