''' TWIP Robot

Ethan Lew
4/18/19
elew@pdx.edu

Model complete robot dynamics, including controllers
'''

from .pid import IterPID
from .motor import PWMDCMotor
from .twip import TWIPZi
from .system import SysBase
import json


class PIDRobot(SysBase):
    ''' PIDRobot

    Represents a twip robot with twip dynamics, two PWM DC motors and two PID controllers.
    '''
    def __init__(self, Ts):
        self.sp_tilt = 0
        self.sp_yaw = 0

        self.twip = TWIPZi()
        self.equations = self.twip.equations

        self.motor_l = PWMDCMotor()
        self.motor_r = PWMDCMotor()
        self.motor_l.set_IC([0, 0])
        self.motor_r.set_IC([0, 0])

        self.pid_tilt = IterPID(Ts)
        self.pid_yaw = IterPID(Ts)

        self.pid_tilt.tune(70, 5, 100)
        self.pid_yaw.tune(25, 2, 0)
        
        self.pid_tilt.set_IC([0, 0, 0, 0])
        self.pid_yaw.set_IC([0, 0, 0, 0])

        self.parameters = self.twip.parameters

    def update_current_state(self, dt, F =None):
        # get twip state
        coords = self.twip.get_position_coordinates()
        curr_yaw = coords[2]
        curr_tilt = coords[5]
        
        # Get error signals
        err_y = self.sp_yaw - curr_yaw
        err_t = self.sp_tilt - curr_tilt

        # Update PID
        self.pid_yaw.update_current_state(dt, [err_y])
        self.pid_tilt.update_current_state(dt, [err_t])

        # Get PID values
        ctrl_t = self.pid_tilt.get_position_coordinates()
        ctrl_y = self.pid_yaw.get_position_coordinates()

        # Convert to PWM values
        pwm_t = int(ctrl_t)
        pwm_y = int(ctrl_y)

        # Update Motors
        self.motor_l.update_current_state(dt, [-pwm_t + pwm_y])
        self.motor_r.update_current_state(dt, [-pwm_t - pwm_y])

        # Get motor torques
        t_l = self.motor_l.get_position_coordinates()[0]
        t_r = self.motor_r.get_position_coordinates()[0]

        tF = [t_l*10, t_r*10,  0, 0]

        if F is not None:
            tF = [(tF[i] + F[i]) for i in range(0, len(F))] 

        # Update TWIP
        self.twip.update_current_state(dt, tF)

    def set_tilt(self, tilt):
        self.sp_tilt = tilt

    def set_yaw(self, yaw):
        self.sp_yaw = yaw

    def set_IC(self, coord):
        self.twip.set_IC(coord)

    def get_position_coordinates(self):
        return self.twip.get_position_coordinates()

    def __str__(self):
        s = '--- TWIP\n'
        s += self.twip.__str__()
        s += '--- Left Motor\n'
        s += self.motor_l.__str__()
        s += '--- Right Motor\n'
        s += self.motor_r.__str__()
        s += '--- Tilt Controller\n'
        s += self.pid_tilt.__str__()
        s += '--- Yaw Controller\n'
        s += self.pid_yaw.__str__()
        return s


def set_params(model, descr, param):
    ''' set_params
    For a loaded dictionary of params, set the parameters in the model object
    :param model: PIDRobot object
    :param descr: robot description
    :param param: parameter name
    :return None
    '''
    if param in descr:
        for p in descr[param]:
            model.set_parameter(p, descr[param][p])


def set_controller(model, descr, param):
    ''' set_controller
    For a loaded dictionary of params, set the parameters in the model's controller
    :param model: PIDRobot object
    :param descr: robot description
    :param param: controller name
    :return None
    '''
    if param in descr:
        if descr[param]["type"] == "PID":
            for p in descr[param]["params"]:
                model.set_parameter(p, descr[param]["params"][p])


def load_robot_json(filename):
    ''' load_robot_json
    
    Load a robot description (stored as a json file) as a PIDRobot object.
    The structure of the file is a follows:
    {
        "twip" : {},
        "motor_left" : {},
        "motor_right" : {},
        "controller_tilt" : {},
        "controller_yaw" : {}
    }

    The params can be found in the respective model documentations themselves.
    TODO: Make a store_robot_json function

    :param filename: name of the json file to load
    :return PIDRobot object
    '''
    rbt = PIDRobot(0.02)
    with open(filename, 'r') as fp:
        rbt_descr = json.load(fp)
        set_params(rbt.twip, rbt_descr, 'twip')
        set_params(rbt.motor_l, rbt_descr, 'motor_left')
        set_params(rbt.motor_l, rbt_descr, 'motor_right')
        set_controller(rbt.pid_tilt, rbt_descr, 'controller_tilt')
        set_controller(rbt.pid_yaw, rbt_descr, 'controller_yaw')
    return rbt

