#! /bin/env python3

''' Send N robots to random waypoints in a MxM space.
At the moment these are sequential, and collisions are avoided
by checking where other robots are (not where they might be going).
'''

import rclpy
from rclpy.node import Node

from freyja_msgs.msg import ReferenceState, CurrentState, WaypointTarget, RLStatesActions
from geometry_msgs.msg import Vector3

import os, sys, time
import math, random
import numpy as np

class Robot:
    def __init__(self, name):
        self.name = name;
        self.action = np.array([np.nan, np.nan, np.nan]);   # [vN, vE, yawR]
        self.prev_action = np.array([np.nan, np.nan, np.nan]);
        self.state = np.array([np.nan]*6);
        self.prev_state = np.array([np.nan]*6);
        self.prev_state_action_updated = False;
        

    def state_callback(self, msg):
        # extract only north, east and yaw components
        self.state = np.array([msg.state_vector[0], msg.state_vector[1], msg.state_vector[8], msg.state_vector[3], msg.state_vector[4], 0.0]);

    @property
    def get_pose(self):
        return self.state[0:3];

    @property
    def get_pos_as_np(self):
        return np.array([self.state[0], self.state[1]]);
        
    @property
    def get_vel(self):
        return self.state[3:6];
    
    def get_current_state(self):
        return self.state;

    def get_previous_state(self):
        return self.prev_state;

    def get_previous_action(self):
        return self.prev_action;

    @property
    def get_current_action(self):
        return self.action;
    
    def set_current_action(self, a):
        self.action = np.array([a[0], a[1], 0.0]);

    def update_previous_action(self):
        self.prev_action = self.action.copy();
        self.prev_state_action_updated = True;
    
    def update_previous_state(self):
        self.prev_state = self.state.copy();
        self.prev_state_action_updated = True;

    @property
    def have_prev_states(self):
        return self.prev_state_action_updated;

class RandomExplore(Node):
    def __init__(self):
        super().__init__("random_explorer");

        self.max_time = 900;  # seconds, expt duration
        self.max_wpts = 20;   # number of waypoints to send each robot to
        self.time_per_wpt = 1.5;
        self.use_sticky_actions = True;

        self.robot_names = ["robomaster_1"];
        self.n_robots = len(self.robot_names);

        # define actions as 8 directions, and a (0,0) direction
        rmat = lambda a : np.array([[np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]]);
        self.actions = [np.dot(rmat(ang), [1.0,0.0]) for ang in np.radians(np.linspace(0, 360-45, 8))];
        self.actions.append(np.array([0,0]));
        self.n_actions = len(self.actions);

        self.fixed_speed = 0.3;
        self.fixed_yaw = 0.0;

        self.robots = [None]*self.n_robots;
        self.wpt_pubs = [None]*self.n_robots;
        self.pos_subs = [None]*self.n_robots;
        self.tuple_pubs = [None]*self.n_robots;
        for idx in range(self.n_robots):
            name = self.robot_names[idx];
            self.robots[idx] = Robot(name);
            self.wpt_pubs[idx] = self.create_publisher(WaypointTarget, name+"/discrete_waypoint_target", 1);
            self.pos_subs[idx] = self.create_subscription(CurrentState, name+"/current_state", self.robots[idx].state_callback, 1);
            self.tuple_pubs[idx] = self.create_publisher(RLStatesActions, name+"/rl_statesactions_tuple", 1);
            print("Initialised: ", name);

        # set limits
        self.northings = [-2.0, 2.0];
        self.eastings = [-2.5, 2.5];
        self.yaws = [0, math.pi*2];
        self.speeds = [0.1, 1.0];
        self.safety_dist = 0.45;

        # define some containers to reuse
        self.wpt_tgt = WaypointTarget();
        self.wpt_tgt.waypoint_mode = 1;   # enum for 'speed' mode

        self.mytime = 0.0;                  # my timekeeping
        self.last_wpt_time = -10.0;         # set large negative to initialise
        self.last_log_time = 0.0;           # helps prevent nan logging
        self.wpt_interval = self.time_per_wpt - 0.5;
        self.logging_interval = 1.0/3.0;
        self.timer_dt = 1.0/100.0;
        self.cmd_timer = self.create_timer(self.timer_dt, self.commander_timer_cb);
        
        # RL message
        self.StAcTuple = RLStatesActions();

    def is_colliding(self, r, pos_n, pos_e):
        # return true if r is likely to collide with anything on
        # its way to (pos_n, pos_e).
        p_orig = self.robots[r].get_pos_as_np;
        p_dest = np.array([pos_n, pos_e]);

        for idx in range(self.n_robots):
            if idx is not r:
                p_other = self.robots[idx].get_pos_as_np;
                d = np.cross(p_dest-p_orig, p_other-p_orig)/np.linalg.norm(p_dest-p_orig);
                if d < self.safety_dist:
                    return True;
        return False;

    def is_withinbounds(self, r, pos):
    	# return true if next pos is within bounds
    	return  (self.northings[0] < pos[0] < self.northings[1]) and (self.eastings[0] < pos[1] < self.eastings[1]);

    def get_as_geomvec3(self, v):
        return Vector3(x=v[0], y=v[1], z=v[2]);
    
    def send_new_waypoint(self, r):
        cur_pos = self.robots[r].get_pos_as_np;
        
        # pick a random index of directions action
        a = random.randint(0, self.n_actions-1);                    # direction
        spd = self.fixed_speed;   #random.uniform( self.speeds[0], self.speeds[1] );     # magnitude
        action = spd*self.actions[a];                               # 2d vector
        if self.use_sticky_actions and random.uniform(0, 1) < 0.5:
            action = self.robots[r].get_current_action[0:2];
        next_aimpos = cur_pos + (action*self.time_per_wpt);         # aim at this point
        next_virtpos = cur_pos + (action*self.wpt_interval);        # likely end up here before next loop
        print("Trying move: ", cur_pos, next_virtpos);
        dest_safe = self.is_withinbounds( r, next_virtpos );
        if not dest_safe:
            next_pos = cur_pos;         # robot control will ignore such a command
            spd = 0.0;                  # redundant
        else:
            next_pos = next_aimpos;
        
        # send command to robot
        self.wpt_tgt.terminal_pn = next_pos[0];
        self.wpt_tgt.terminal_pe = next_pos[1];
        self.wpt_tgt.terminal_yaw = self.fixed_yaw;
        self.wpt_tgt.translational_speed = spd;
        self.wpt_tgt.header.stamp = self.get_clock().now().to_msg();
        self.wpt_pubs[r].publish( self.wpt_tgt );

        self.robots[r].set_current_action(action);

    def log_rl_tuples(self, r):

        if not self.robots[r].have_prev_states:
            self.robots[r].update_previous_state();
            self.robots[r].update_previous_action();
            return;
        
        # snapshots as of now
        cur_state = self.robots[r].get_current_state();
        prev_state = self.robots[r].get_previous_state();
        prev_action = self.robots[r].get_previous_action();
        
        # publish
        self.StAcTuple.header.stamp = self.get_clock().now().to_msg();
        self.StAcTuple.prev_state_pos = self.get_as_geomvec3(prev_state[0:3]);
        self.StAcTuple.prev_state_vel = self.get_as_geomvec3(prev_state[3:6]);
        self.StAcTuple.curr_state_pos = self.get_as_geomvec3(cur_state[0:3]);
        self.StAcTuple.curr_state_vel = self.get_as_geomvec3(cur_state[3:6]);
        self.StAcTuple.prev_action = self.get_as_geomvec3(prev_action);
        self.tuple_pubs[r].publish( self.StAcTuple );

        # shuffle back to make current the 'previous' for next time
        self.robots[r].update_previous_state();
        self.robots[r].update_previous_action();

    
    def commander_timer_cb(self):
        # pick a random robot and send it somewhere, or just code the particular robot idx in `r`
        r = 0;   # random.randint(0, self.n_robots-1);

        if (self.mytime - self.last_wpt_time) >= self.wpt_interval :
            self.send_new_waypoint(r);
            self.last_wpt_time = self.mytime;
            print( "Time: ", self.mytime, "/", self.max_time );

        if (self.mytime - self.last_log_time) >= self.logging_interval :
            self.log_rl_tuples(r);
            self.last_log_time = self.mytime;

        # book-keeping
        self.mytime += self.timer_dt;

        if self.mytime > self.max_time:
            rclpy.shutdown();
            sys.exit(0);



def main():
    rclpy.init();
    explorer_node = RandomExplore();

    rclpy.spin(explorer_node);

    explorer_node.destroy_node();
    rclpy.shutdown();

if __name__ == '__main__':
    main();
