# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(1,2), (2,1),(1,3), (3,1),(1,4), (4,1),(1,5), (5,1),(2,3), (3,2),(2,4), (4,2),(2,5), (5,2),
                            (3,4), (4,3),(3,5), (5,3),(4,5), (5,4),(0,0)]
        self.state_space =  [(a, b, c) for a in range(1, m+1) for b in range(t) for c in range(d)]
        self.state_init =   random.choice([(1,0,0), (2,0,0), (3,0,0), (4,0,0), (5,0,0)])

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""

        state_encod = [0] * (m + t + d)

        # location
        state_encod[state[0] - 1] = 1

        # hour
        state_encod[m + state[1]] = 1

        # day
        state_encod[m + t + state[2]] = 1

        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        print(location)
        if location == 0:
            requests = np.random.poisson(2)
        if location == 2:
            requests = np.random.poisson(12)
        if location == 3:
            requests = np.random.poisson(4)
        if location == 4:
            requests = np.random.poisson(7)
        if location == 5:
            requests = np.random.poisson(8)

        if requests > 15:
            requests = 15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests)
        actions = [self.action_space[i] for i in possible_actions_index]
        actions.append([0,0])

        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        start_loc, time, day = state
        pickup, drop = action


        if action == [0, 0]:
            return -C
        else:
            time_elapsed_till_pickup = Time_matrix[start_loc, pickup, time, day]
            time_next = np.int((time + time_elapsed_till_pickup) % t)
            day_next = np.int((day + (time + time_elapsed_till_pickup) // t) % d)
            timepicktodrop = Time_matrix[pickup, drop, time_next, day_next]
            r_cost = R * timepicktodrop
            c_cost = C * (timepicktodrop + Time_matrix[start_loc, pickup, time, day])
            reward = r_cost - c_cost
            return reward



def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        next_state = []

        total_time = 0
        transit_time = 0
        wait_time = 0
        ride_time = 0

        curr_loc = self.state_get_loc(state)
        pickup_loc = self.action_get_pickup(action)
        drop_loc = self.action_get_drop(action)
        curr_time = self.state_get_time(state)
        curr_day = self.state_get_day(state)

        if ((pickup_loc == 0) and (drop_loc == 0)):
            wait_time = 1
            next_loc = curr_loc
        elif (curr_loc == pickup_loc):
            ride_time = Time_matrix[curr_loc][drop_loc][curr_time][curr_day]
            next_loc = drop_loc
        else:
            transit_time = Time_matrix[curr_loc][pickup_loc][curr_time][curr_day]
            new_time, new_day = self.update_time(curr_time, curr_day, transit_time)
            ride_time = Time_matrix[pickup_loc][drop_loc][new_time][new_day]
            next_loc = drop_loc

        total_time = (wait_time + transit_time + ride_time)
        next_time, next_day = self.update_time(curr_time, curr_day, total_time)

        next_state = [next_loc, next_time, next_day]

        return next_state

    def update_time(self, time, day, ride):
        ride = int(ride)
        if (time + ride) < 24:
            time = time + ride
        else:
            time = (time + ride) % 24
            num_days = (time + ride) // 24
            day = (day + num_days) % 7
        return time, day


    # all getters & setters are declared here
    def state_get_loc(self, state):
        return state[0]

    def state_get_time(self, state):
        return state[1]

    def state_get_day(self, state):
        return state[2]

    def action_get_pickup(self, action):
        return action[0]

    def action_get_drop(self, action):
        return action[1]

    def state_set_loc(self, state, loc):
        state[0] = loc

    def state_set_time(self, state, time):
        state[1] = time

    def state_set_day(self, state, day):
        state[2] = day

    def action_set_pickup(self, action, pickup):
        action[0] = pickup

    def action_set_drop(self, action, drop):
        action[1] = drop

    def reset(self):
        """Return the current state and action space"""
        return self.action_space, self.state_space, self.state_init
