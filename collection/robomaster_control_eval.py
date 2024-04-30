import rclpy
from rclpy.node import Node

from freyja_msgs.msg import ReferenceState, CurrentState

import robomaster_control as RControl

import os, sys, time
import math, random
import numpy as np

class Robot:
    def __init__(self, name):
        self.name = name;
        self.state = ReferenceState();      # abuse this struct to use named elements
        self.have_state = False;

    def state_callback(self, msg):
        # extract only specific components
        self.state.pn = msg.state_vector[0];
        self.state.pe = msg.state_vector[1];
        self.state.vn = msg.state_vector[3];
        self.state.ve =  msg.state_vector[4];
        self.state.yaw = msg.state_vector[8];
        self.have_state = True;

    @property
    def get_pos_as_np(self):
        return np.array([self.state.pn, self.state.pe]);

    def get_state(self):
        return self.state;

    @property
    def ready(self):
        return self.have_state;

class RobomasterEval(Node):
    def __init__(self):
        super().__init__("robomaster_eval");

        self.max_time = 600;  # seconds, expt duration
        self.robot_names = ["robomaster_1"];
        self.n_robots = len(self.robot_names);

        ### define actions as 8 directions, and a (0,0) direction
        # rmat = lambda a : np.array([[np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]]);
        # self.actions = [np.dot(rmat(ang), [1.0,0.0]) for ang in np.radians(np.linspace(0, 360-45, 8))];
        # self.actions.append(np.array([0,0]));
        # self.n_actions = len(self.actions);

        self.robots = [None]*self.n_robots;
        self.state_subs = [None]*self.n_robots;
        self.ref_pubs = [None]*self.n_robots;
        for idx in range(self.n_robots):
            name = self.robot_names[idx];
            self.robots[idx] = Robot(name);
            self.state_subs[idx] = self.create_subscription(CurrentState, name+"/current_state", self.robots[idx].state_callback, 1);
            self.ref_pubs[idx] = self.create_publisher(ReferenceState, name+"/reference_state", 1);
            print("Initialised: ", name);

        # set limits
        self.northings = [-2.0, 2.0];
        self.eastings = [-2.5, 2.5];
        self.yaws = [0, math.pi*2];
        self.speeds = [0.1, 1.0];
        self.safety_dist = 0.45;

        self.RefState = ReferenceState();
    
        self.print_help_once = True;
        self.mytime = 0.0;
        freq = 1.0;                 # Hz
        self.timer_dt = 1.0/freq;
        self.cmd_timer = self.create_timer(self.timer_dt, self.commander_timer_cb);

    def commander_timer_cb(self):
        # pick a random robot and send it somewhere, or just code the particular robot idx in `r`
        r = 0;   # random.randint(0, self.n_robots-1);
        #print(RControl.mappings.keys());
        if not self.robots[r].ready:
            print("Waiting for state..");
            if self.print_help_once:
                print("Av. keys: ", RControl.mappings.keys());
                self.print_help_once = False;
            return;
        
        state = self.robots[r].get_state();                                       # returns struct type
        state = np.array([state.pe, state.pn, state.yaw, state.ve, state.vn]);    # note the order
        prompt_str = list(RControl.mappings.keys())[1];
        act = RControl.policy_wrapper( state, RControl.mappings[ prompt_str ] );

        self.RefState.ve = float(act[0]);
        self.RefState.vn = float(act[1]);
        self.ref_pubs[r].publish(self.RefState);

        print( f"prompt_str: {prompt_str}, action (ve/vn): {act[0]:.2f}/{act[1]:.2f}, state (pe/pn): {state[1]:.2f}/{state[0]:.2f}" );

        # book-keeping
        self.mytime += self.timer_dt;
        if self.mytime > self.max_time:
            rclpy.shutdown();
            sys.exit(0);

def main():
    rclpy.init();
    eval_node = RobomasterEval();

    rclpy.spin(eval_node);

    eval_node.destroy_node();
    rclpy.shutdown();

if __name__ == '__main__':
    main();