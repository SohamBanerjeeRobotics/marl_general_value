from constants import ACTION_MAPPING, ARENA_BOUNDS_N, ARENA_BOUNDS_E, ACTION_IDX, ROBOT_DIAMETER
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32

from freyja_msgs.msg import ReferenceState, CurrentState

from rewards import segment_collision
import robomaster_control as RControl

import os, sys, time
import math, random
import numpy as np
import pandas as pd
import time

random.seed(0)
np.random.seed(0)
mappings = RControl.eval_mappings # Can be train or eval


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
    ros_timestamps = []

    def setup(self):
        self.max_time = 60 * 10;  # seconds, expt duration
        #self.robot_names = ["robomaster_0", "robomaster_1", "robomaster_2", "robomaster_3", "robomaster_4"]
        self.robot_names = ["robomaster_2"]
        self.n_robots = len(self.robot_names)

        self.robots = [None]*self.n_robots
        self.state_subs = [None]*self.n_robots
        self.ref_pubs = [None]*self.n_robots
        self.goal_pubs = [None]*self.n_robots
        self.string_pubs = [None]*self.n_robots
        self.dist2goal_pubs = [None]*self.n_robots
        for idx in range(self.n_robots):
            name = self.robot_names[idx]
            self.robots[idx] = Robot(name)
            self.state_subs[idx] = self.create_subscription(CurrentState, name+"/current_state", self.robots[idx].state_callback, 1)
            self.ref_pubs[idx] = self.create_publisher(ReferenceState, name+"/reference_state", 1)
            self.goal_pubs[idx] = self.create_publisher(ReferenceState, name+"/goal_state", 1)
            self.dist2goal_pubs[idx] = self.create_publisher(Float32, name+"/dist_to_goal", 1)
            self.string_pubs[idx] = self.create_publisher(String, name+"/goal_string", 1)
            print("Initialised: ", name)

        self.print_help_once = True;
        self.mytime = 0.0
        freq = 1.0                 # Hz
        self.timer_dt = 1.0/freq
        self.cmd_timer = self.create_timer(self.timer_dt, self.commander_timer_cb)

    def teardown(self):
        pass

    def get_action_idx(self, state):
        raise NotImplementedError()

    def get_safe_action(self, state, action_idx):
        pos = state[:, :2]
        action_vel = np.stack([ACTION_MAPPING[idx.item()] for idx in action_idx], axis=0)
        next_pos = state[:, :2] + action_vel * self.timer_dt
        # if collision, return null action
        collision = segment_collision(pos, next_pos, ROBOT_DIAMETER)
        # from jax array to numpy for assignment
        action_idx = np.array(action_idx)
        action_idx[collision.astype(bool)] = ACTION_IDX["0"]
        if np.any(collision):
            print("Collision detected: ", collision)
        return action_idx


    def commander_timer_cb(self):
        for r in range(len(self.robots)):
            if not self.robots[r].ready:
                print(f"Waiting for state for robot {self.robot_names[r]}..")
                return

        states = [
            np.array([r.state.pn, r.state.pe, r.state.vn, r.state.ve])
            for r in self.robots
        ]
        states = np.stack(states, axis=0)
        self.states.append(states)

        action_idx = self.get_action_idx(states)
        #action_idx = self.get_safe_action(states, action_idx)
        action_vel = np.stack([ACTION_MAPPING[idx.item()] for idx in action_idx], axis=0)
        self.actions.append(action_idx)
        self.action_vels.append(action_vel)

        action_str = [list(ACTION_IDX.keys())[idx.item()] for idx in action_idx]
        print(
            f"{self.mytime}/{self.max_time}s\n"
            #f"action: {ACTION_MAPPING[action_idx]}"
            f"action: {action_str}\n"
            #f"state (pn/pe/ve/vn): {state[0]:.2f}/{state[1]:.2f}/{state[2]:.2f}/{state[3]:.2f}" 
            #f"state (pn/pe/ve/vn): {states}"
        )

        self.ros_timestamps.append(self.get_clock().now().nanoseconds)
        for r in range(len(self.robots)):
            rstate = ReferenceState()
            rstate.vn = action_vel[r, 0].item()
            rstate.ve = action_vel[r, 1].item()
            rstate.header.stamp = self.get_clock().now().to_msg()
            self.ref_pubs[r].publish(rstate)

        # book-keeping
        self.mytime += self.timer_dt
        if self.mytime > self.max_time:
            for p in self.ref_pubs:
                rstate = ReferenceState()
                p.publish(rstate)
            self.teardown()
            rclpy.shutdown()

        
class RoboMasterCollect(RoboMasterBase):
    def __init__(self):
        super().__init__("robomaster_collect")
        self.setup()


    def get_action_idx(self, state):
        #action = random.randint(0, len(ACTION_MAPPING) - 1)
        action = np.random.randint(0, len(ACTION_MAPPING) - 1, len(self.robots))
        self.action = action

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


class RoboMasterEval(RoboMasterBase):
    goals = []
    key = RControl.key

    def __init__(self):
        super().__init__("robomaster_eval")
        self.setup()
        self.task_idx = np.random.choice(len(mappings), len(self.robots), replace=False)
        self.task_string_idx = [0] * len(self.robots)
        self.task_string_idx = np.random.randint(
            [len(mappings[idx]) for idx in self.task_idx]
        )
        self.task_timer = 0
        self.max_task_time = 30

    def get_action_idx(self, state):
        tasks = [mappings[idx] for idx in self.task_idx]
        # First index is the robot idx, second is the task idx
        prompt_strs = [tasks[i][idx]['task_string'] for i, idx in enumerate(self.task_string_idx)]
        embeddings = np.stack(
            [tasks[i][idx]['embedding'] for i, idx in enumerate(self.task_string_idx)], axis=0
        )
        goals = np.stack(
            [tasks[i][idx]['goal'] for i, idx in enumerate(self.task_string_idx)], axis=0
        )

        action_idx, self.key = RControl.ma_policy_wrapper(
            RControl.ma_q_function,
            state, 
            embeddings,
            self.key
        )

        # Log goals for posterity
        self.goals.append(goals)
        goal_dists = np.linalg.norm(goals - state[:, :2], axis=-1)
        for i in range(self.n_robots):
            goal_msg = ReferenceState()
            goal_msg.pn, goal_msg.pe = goals[i, 0], goals[i, 1]
            goal_msg.header.stamp = self.get_clock().now().to_msg()
            self.goal_pubs[i].publish(goal_msg)

            str_msg = String()
            str_msg.data = prompt_strs[i]
            self.string_pubs[i].publish(str_msg)

            dist = Float32()
            dist.data = goal_dists[i]
            self.dist2goal_pubs[i].publish(dist)

        print(f"Mean dist. {np.mean(goal_dists)}")

        self.task_timer += 1
        if self.task_timer >= self.max_task_time:
            print("Next task!")
            self.task_idx = np.random.choice(len(mappings), len(self.robots), replace=False)
            tasks = [mappings[idx] for idx in self.task_idx]
            self.task_string_idx = np.random.randint(
                [len(tasks[i]) for i in range(self.n_robots)]
            )
            self.task_timer = 0
        print(
                f"Task: {prompt_strs} ({self.task_timer:.0f}/{self.max_task_time:.0f}s)"
        )
        return action_idx

    def teardown(self):
        dfs = []
        state = np.stack(self.states[:-1])
        next_state = np.stack(self.states[1:])
        action = np.stack(self.actions[:-1])
        action_vels = np.stack(self.action_vels[:-1])
        goals = np.stack(self.goals[:-1])
        for idx, name in enumerate(self.robot_names):
            s = pd.DataFrame.from_dict({
                "state.pn": state[:, idx, 0],
                "state.pe": state[:, idx, 1],
                "state.vn": state[:, idx, 2],
                "state.ve": state[:, idx, 3],
            })
            next_s = pd.DataFrame.from_dict({
                "next_state.pn": next_state[:, idx, 0],
                "next_state.pe": next_state[:, idx, 1],
                "next_state.vn": next_state[:, idx, 2],
                "next_state.ve": next_state[:, idx, 3],
            })
            v = pd.DataFrame.from_dict({
                "action_vel.vn": action_vels[:, idx, 0],
                "action_vel.ve": action_vels[:, idx, 1],
            })
            g = pd.DataFrame.from_dict({
                "goals.pn": goals[:, idx, 0],
                "goals.pe": goals[:, idx, 1],
            })
            df = pd.DataFrame({
                **s,
                **next_s,
                **v,
                **g,
                "action": action[:, idx],
                "ros_timestamp_ns": self.ros_timestamps[:-1],
            })
            df['robot_name'] = name
            df['robot_idx'] = idx
            dfs.append(df)
        df = pd.concat(dfs).reset_index(drop=True)
        print('Mean dist to goal', np.linalg.norm(state[:, :, :2] - goals, axis=-1).mean())
        df.to_csv(f"data/robomaster_eval_{int(time.time())}.csv")
