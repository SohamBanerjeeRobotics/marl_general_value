from constants import ACTION_MAPPING, ARENA_BOUNDS_N, ARENA_BOUNDS_E, ACTION_IDX
import rclpy
from rclpy.node import Node

from freyja_msgs.msg import ReferenceState, CurrentState

import robomaster_control as RControl

import os, sys, time
import math, random
import numpy as np
import pandas as pd
import time


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
        self.have_state = True;

    @property
    def ready(self):
        return self.have_state;


class RoboMasterBase(Node):
    action = None
    states = []
    actions = []
    action_vels = []

    def setup(self):
        self.max_time = 900;  # seconds, expt duration
        self.robot_names = ["robomaster_1"]
        self.n_robots = len(self.robot_names)

        self.robots = [None]*self.n_robots
        self.state_subs = [None]*self.n_robots
        self.ref_pubs = [None]*self.n_robots
        for idx in range(self.n_robots):
            name = self.robot_names[idx]
            self.robots[idx] = Robot(name)
            self.state_subs[idx] = self.create_subscription(CurrentState, name+"/current_state", self.robots[idx].state_callback, 1)
            self.ref_pubs[idx] = self.create_publisher(ReferenceState, name+"/reference_state", 1)
            print("Initialised: ", name)

        self.RefState = ReferenceState();
    
        self.print_help_once = True;
        self.mytime = 0.0
        freq = 1.0                 # Hz
        self.timer_dt = 1.0/freq
        self.cmd_timer = self.create_timer(self.timer_dt, self.commander_timer_cb)

    def teardown(self):
        pass

    def get_action_idx(self, state):
        raise NotImplementedError()


    def safe_action(self, state, action_idx):
        if state[0] < ARENA_BOUNDS_N[0] and action_idx in [ACTION_IDX["S"], ACTION_IDX["SW"], ACTION_IDX["SE"]]:
            return False
        elif state[0] > ARENA_BOUNDS_N[1] and action_idx in [ACTION_IDX["N"], ACTION_IDX["NW"], ACTION_IDX["NE"]]:
            return False
        elif state[1] < ARENA_BOUNDS_E[0] and action_idx in [ACTION_IDX["W"], ACTION_IDX["NW"], ACTION_IDX["SW"]]:
            return False
        elif state[1] > ARENA_BOUNDS_E[1] and action_idx in [ACTION_IDX["E"], ACTION_IDX["NE"], ACTION_IDX["SE"]]:
            return False
        else:
            return True

    def commander_timer_cb(self):
        r = 0
        if not self.robots[r].ready:
            print("Waiting for state..")
            if self.print_help_once:
                print("Av. keys: ", RControl.mappings.keys())
                self.print_help_once = False
            return

        state = self.robots[r].state
        state = np.array([state.pn, state.pe, state.vn, state.ve])
        self.states.append(state)

        action_idx = self.get_action_idx(state)
        action_vel = ACTION_MAPPING[action_idx]
        self.actions.append(action_idx)
        self.action_vels.append(action_vel)

        action_str = list(ACTION_IDX.keys())[action_idx]
        print(
            f"{self.mytime}/{self.max_time}s\n"
            f"action: {ACTION_MAPPING[action_idx]}"
            f"/{action_str}\n"
            f"state (pn/pe/ve/vn): {state[0]:.2f}/{state[1]:.2f}/{state[2]:.2f}/{state[3]:.2f}" 
        )

        self.RefState.vn = action_vel[0].item()
        self.RefState.ve = action_vel[1].item()
        self.ref_pubs[r].publish(self.RefState)

        # book-keeping
        self.mytime += self.timer_dt
        if self.mytime > self.max_time:
            self.RefState.vn = 0.0
            self.RefState.ve = 0.0
            self.ref_pubs[r].publish(self.RefState)
            self.teardown()
            rclpy.shutdown()

        
class RoboMasterCollect(RoboMasterBase):
    def __init__(self):
        super().__init__("robomaster_collect")
        self.setup()


    def get_action_idx(self, state):
        action = random.randint(0, len(ACTION_MAPPING) - 1)
        self.action = action

        if not self.safe_action(state, action):
            return self.get_action_idx(state)
        
        return action

    def teardown(self):
        state = np.stack(self.states[:-1])
        next_state = np.stack(self.states[1:])
        action = np.stack(self.actions[:-1])
        action_vels = np.stack(self.action_vels[:-1])
        state = pd.DataFrame.from_dict({
            "state.pn": state[:,0],
            "state.pe": state[:,1],
            "state.vn": state[:,2],
            "state.ve": state[:,3],
        })
        next_state = pd.DataFrame.from_dict({
            "next_state.pn": next_state[:,0],
            "next_state.pe": next_state[:,1],
            "next_state.vn": next_state[:,2],
            "next_state.ve": next_state[:,3],
        })
        action_vels = pd.DataFrame.from_dict({
            "action_vel.vn": action_vels[:,0],
            "action_vel.ve": action_vels[:,1],
        })
        df = pd.DataFrame({
            **state,
            **next_state,
            **action_vels,
            "action": action,
        })
        df.to_csv(f"data/robomaster_collect_{int(time.time())}.csv")


class RobomasterEval(RoboMasterBase):
    def __init__(self):
        super().__init__("robomaster_eval")
        self.setup()
        self.task_idx = 0

    def get_action_idx(self, state):
        prompt_str = list(RControl.mappings.keys())[self.task_idx];
        embedding, goal = RControl.mappings[prompt_str]['embedding'], RControl.mappings[prompt_str]['goal']

        if np.linalg.norm(state[:2] - goal) < 0.3:
            print("Completed task!")
            self.task_idx = (self.task_idx + 1) % len(RControl.mappings)
            prompt_str = list(RControl.mappings.keys())[self.task_idx]
            embedding, goal = RControl.mappings[prompt_str]['embedding'], RControl.mappings[prompt_str]['goal']

        action_idx = RControl.policy_wrapper(
            RControl.q_function,
            state, 
            embedding,
        )
        action_str = list(RControl.ACTION_IDX.keys())[action_idx]
        print(
            f"Task: {self.task_idx}"
            f"prompt_str: {prompt_str}\n"
            f"action: {ACTION_MAPPING[action_idx]}"
            f"/{action_str}\n"
            f"state (pn/pe/ve/vn): {state[0]:.2f}/{state[1]:.2f}/{state[2]:.2f}/{state[3]:.2f}" 
        )
        return action_idx

    def teardown(self):
        df = pd.DataFrame.from_dict({
            "state": np.stack(self.states[:-1]),
            "next_state": np.stack(self.states[1:]),
            "action": np.stack(self.actions[:-1]),
            "action_vel": np.stack(self.action_vels[:-1])
        })
        df.to_csv(f"data/robomaster_eval_{int(time.time())}.csv")

def main():
    rclpy.init();
    eval_node = RobomasterEval();

    rclpy.spin(eval_node);

    eval_node.destroy_node();
    rclpy.shutdown();

if __name__ == '__main__':
    main();
