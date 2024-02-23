import gymnasium as gym
#from gym import spaces
import pygame
import numpy as np
import pandas as pd
from stable_baselines3.common.env_checker import check_env
from copy import deepcopy
from random import randint
from math import log2, ceil, floor


#------------------------------------Environment Class----------------------------------------------------
class SliceManagementEnv1(gym.Env):
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        # Define environment parameters

        # RAN Global Parameters -------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.numerology = 1                       # 0,1,2,3,...
        self.scs = 2**(self.numerology) * 15_000   # Hz
        self.slot_per_subframe = 2**(self.numerology)
        
        self.channel_BW = 10_000_000              # Hz (100MHz for <6GHz band, and 400MHZ for mmWave)
        self.guard_BW = 845_000                   # Hz (for symmetric guard band)

        self.PRB_BW = self.scs * 12               # Hz - Bandwidth for one PRB (one OFDM symbol, 12 subcarriers)
        self.PRB_per_channel = floor((self.channel_BW - (2*self.guard_BW)) / (self.PRB_BW))        # Number of PRB to allocate within the channel bandwidth
        self.sprectral_efficiency = 5.1152        # bps/Hz (For 64-QAM and 873/1024)
        
        #Defined parameters per Slice. (Each component is a list of the correspondent slice parameters)-------------------------------------------------------------------------------------
        #self.slices_param = [10, 20, 50]
        self.slices_param = {1: [4, 16, 100, 40, 50], 2: [4, 32, 100, 100, 30], 3: [8, 16, 32, 80, 20], 
                             4: [4, 8, 16, 50, 25], 5: [2, 8, 32, 40, 10], 6: [2, 8, 32, 40, 5]}

        #self.slice_requests = pd.read_csv('/home/mario/Documents/DQN_Models/Model 1/gym-examples4/gym_examples/slice_request_db4')  # Load VNF requests from the generated CSV
        #self.slice_requests = pd.read_csv('/data/scripts/DQN_models/Model1/gym_examples/slice_request_db1')    #For pod
        
        # VECTORS----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.observation_space = gym.spaces.Box(low=0, high=10000, shape=(2,), dtype=np.float32) #ovservation space composed by Requested resources (MEC BW) and available MEC resources.
        
        self.action_space = gym.spaces.Discrete(3)  # 0: Blocking, 1: Execute Configuration Action , 2: Do Nothing

        #self.process_requests()
        
        # Other necessary variables and data structures
        self.current_time_step = 1
        self.reward = 0
        self.first = True
        
        # Flags -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.config_flag = 0
        self.resources_flag = 1
        

        #--------------------------------------------------------------------------------------Imported variables ---------------------------------------------------------------------------------
        self.processed_requests = []
        self.PRB_map = np.zeros((14, self.PRB_per_channel))                                 # PRB map per slot (14 OFDM symbols x Number of PRB to allocate within the channel bandwidth)

        #Available MEC resources (Order: MEC_CPU (Cores), MEC_RAM (GB), MEC_STORAGE (GB), MEC_BW (Mbps))
        #self.resources = [1000]
        self.resources_1 = {'MEC_CPU': 30, 'MEC_RAM': 128, 'MEC_STORAGE': 100, 'MEC_BW': 300}
        self.resources_2 = {'MEC_CPU': 30, 'MEC_RAM': 128, 'MEC_STORAGE': 100, 'MEC_BW': 200}
        self.resources_3 = {'MEC_CPU': 50, 'MEC_RAM': 128, 'MEC_STORAGE': 100, 'MEC_BW': 200}
        self.resources_4 = {'MEC_CPU': 30, 'MEC_RAM': 128, 'MEC_STORAGE': 100, 'MEC_BW': 200}
        self.resources_5 = {'MEC_CPU': 20, 'MEC_RAM': 64, 'MEC_STORAGE': 80, 'MEC_BW': 100}
        self.resources_6 = {'MEC_CPU': 20, 'MEC_RAM': 64, 'MEC_STORAGE': 80, 'MEC_BW': 80}

    #-------------------------Reset Method----------------------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        # Initialize the environment to its initial state

        #print(self.processed_requests)

        #generate_vnf_list()

        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self.current_time_step = 1
        self.reward = 0
        self.processed_requests = []
        self.reset_resources()
        self.slice_requests = pd.read_csv('/home/mario/Documents/DQN_Models/Model 1/gym-examples4/gym_examples/slice_request_db4')  # Load VNF requests from the generated CSV
        #self.slice_requests = pd.read_csv('/data/scripts/DQN_models/Model1/gym_examples/slice_request_db1')    #For pod
        self.next_request = self.read_request()
        #slice_id = self.create_slice(self.next_request)
        self.update_slice_requests(self.next_request)
        self.check_resources(self.next_request)
        self.observation = np.array([self.next_request['SLICE_MEC_CPU_REQUEST']] + [self.next_request['SLICE_MEC_RAM_REQUEST']] 
                                    + [self.next_request['SLICE_MEC_STORAGE_REQUEST']] + [self.next_request['SLICE_MEC_BW_REQUEST']] 
                                    + [self.next_request['SLICE_RAN_R_REQUEST']]+ [self.resources_flag], dtype=np.float32)
        #self.observation = np.array([self.next_request[1]] + deepcopy(self.resources), dtype=np.float32)
        self.info = {}
        self.first = True
        
        print("\nReset: ", self.observation)
        
        return self.observation, self.info

    #-------------------------Step Method-------------------------------------------------------------------------
    def step(self, action):
        
        if self.first:
            self.next_request = self.processed_requests[0]
            self.first = False
        #else: 
            #next_request = self.read_request()
        #    self.update_slice_requests(self.next_request)
            
        terminated = False
        
        slice_id = self.create_slice(self.next_request)
        
        reward_value = 1
        
        # Apply the selected action (0: Do Nothing, 1: Allocate Slice 1, 2: Allocate Slice 2, 3: Allocate Slice 3)
        terminated = self.evaluate_action(action, slice_id, reward_value, terminated) 

        self.update_slice_requests(self.next_request)

        self.check_resources(self.next_request)
    
        #self.observation = np.array([self.next_request[1]] + [self.resources_flag], dtype=np.float32)
        self.observation = np.array([self.next_request['SLICE_MEC_CPU_REQUEST']] + [self.next_request['SLICE_MEC_RAM_REQUEST']] 
                                    + [self.next_request['SLICE_MEC_STORAGE_REQUEST']] + [self.next_request['SLICE_MEC_BW_REQUEST']] 
                                    + [self.next_request['SLICE_RAN_R_REQUEST']] + [self.resources_flag], dtype=np.float32)
        
        #done = False
        
        info = {}  # Additional information (if needed)
        
        #self.current_time_step += 1  # Increment the time step
        
        #print("Action: ", action, "\nObservation: ", self.observation, "\nReward: ", self.reward)

        truncated = False
        
        return self.observation, self.reward, terminated, truncated, info
    
    def read_request(self):
        next_request = self.slice_requests.iloc[self.current_time_step - 1]

        SiNR = randint(1, 20)

        #request_list =list([next_request['ARRIVAL_REQUEST_@TIME'], next_request['SLICE_BW_REQUEST'], next_request['SLICE_KILL_@TIME']])
        request_list = {'ARRIVAL_REQUEST_@TIME': next_request['ARRIVAL_REQUEST_@TIME'], 'SLICE_MEC_CPU_REQUEST': next_request['SLICE_MEC_CPU_REQUEST'], 
                        'SLICE_MEC_RAM_REQUEST': next_request['SLICE_MEC_RAM_REQUEST'], 'SLICE_MEC_STORAGE_REQUEST': next_request['SLICE_MEC_STORAGE_REQUEST'],
                        'SLICE_MEC_BW_REQUEST': next_request['SLICE_MEC_BW_REQUEST'], 'SLICE_RAN_R_REQUEST': next_request['SLICE_RAN_R_REQUEST'], 
                        'SLICE_KILL_@TIME': next_request['SLICE_KILL_@TIME'], 'UE_ID': self.current_time_step, 'UE_SiNR': SiNR}
        self.current_time_step += 1
        return request_list
        
    def update_slice_requests(self, request):
        # Update the slice request list by deleting the killed VNFs
        if len(self.processed_requests) != 0:
            for i in self.processed_requests:
                #i[2] < request[0]
                if len(i)== 10 and i['SLICE_KILL_@TIME'] <= request['ARRIVAL_REQUEST_@TIME']:
                    slice_id = self.create_slice(i)
                    self.deallocate_slice(i, slice_id)
                    self.processed_requests.remove(i)
        self.processed_requests.append(request)
        
    def check_resources(self, request):
        # Logic to check if there are available resources to allocate the VNF request
        # Set Resources flag to 1 if resources are available, 0 otherwise

        #Check RAN Resources ----------------------------------------------------------------------------------------------------------------------------
        ran_resources = self.check_RAN(request)


        # Check MEC resources----------------------------------------------------------------------------------------------------------------------------
        slice_id = self.create_slice(request)

        if slice_id == 1:
            if (self.resources_1['MEC_CPU'] >= request['SLICE_MEC_CPU_REQUEST'] and self.resources_1['MEC_RAM'] >= request['SLICE_MEC_RAM_REQUEST'] and 
                self.resources_1['MEC_STORAGE'] >= request['SLICE_MEC_STORAGE_REQUEST'] and self.resources_1['MEC_BW'] >= request['SLICE_MEC_BW_REQUEST'] and ran_resources
                ):
                self.resources_flag = 1
            else: self.resources_flag = 0
        elif slice_id == 2:
            if (self.resources_2['MEC_CPU'] >= request['SLICE_MEC_CPU_REQUEST'] and self.resources_2['MEC_RAM'] >= request['SLICE_MEC_RAM_REQUEST'] and 
                self.resources_2['MEC_STORAGE'] >= request['SLICE_MEC_STORAGE_REQUEST'] and self.resources_2['MEC_BW'] >= request['SLICE_MEC_BW_REQUEST'] and ran_resources
                ):
                self.resources_flag = 1
            else: self.resources_flag = 0
        elif slice_id == 3:
            if (self.resources_3['MEC_CPU'] >= request['SLICE_MEC_CPU_REQUEST'] and self.resources_3['MEC_RAM'] >= request['SLICE_MEC_RAM_REQUEST'] and 
                self.resources_3['MEC_STORAGE'] >= request['SLICE_MEC_STORAGE_REQUEST'] and self.resources_3['MEC_BW'] >= request['SLICE_MEC_BW_REQUEST'] and ran_resources
                ):
                self.resources_flag = 1
            else: self.resources_flag = 0
        elif slice_id == 4:
            if (self.resources_4['MEC_CPU'] >= request['SLICE_MEC_CPU_REQUEST'] and self.resources_4['MEC_RAM'] >= request['SLICE_MEC_RAM_REQUEST'] and 
                self.resources_4['MEC_STORAGE'] >= request['SLICE_MEC_STORAGE_REQUEST'] and self.resources_4['MEC_BW'] >= request['SLICE_MEC_BW_REQUEST'] and ran_resources
                ):
                self.resources_flag = 1
            else: self.resources_flag = 0
        elif slice_id == 5:
            if (self.resources_5['MEC_CPU'] >= request['SLICE_MEC_CPU_REQUEST'] and self.resources_5['MEC_RAM'] >= request['SLICE_MEC_RAM_REQUEST'] and 
                self.resources_5['MEC_STORAGE'] >= request['SLICE_MEC_STORAGE_REQUEST'] and self.resources_5['MEC_BW'] >= request['SLICE_MEC_BW_REQUEST'] and ran_resources
                ):
                self.resources_flag = 1
            else: self.resources_flag = 0
        elif slice_id == 6:
            if (self.resources_6['MEC_CPU'] >= request['SLICE_MEC_CPU_REQUEST'] and self.resources_6['MEC_RAM'] >= request['SLICE_MEC_RAM_REQUEST'] and 
                self.resources_6['MEC_STORAGE'] >= request['SLICE_MEC_STORAGE_REQUEST'] and self.resources_6['MEC_BW'] >= request['SLICE_MEC_BW_REQUEST'] and ran_resources
                ):
                self.resources_flag = 1
            else: self.resources_flag = 0
    
    def allocate_slice(self, request, slice_id):
        # Allocate the resources requested by the current VNF

        # RAN resources part---------------------------------------------------------------
        self.allocate_ran(request)
        
        # MEC resources part---------------------------------------------------------------
        if slice_id == 1:
            self.resources_1['MEC_CPU'] -= request['SLICE_MEC_CPU_REQUEST']
            self.resources_1['MEC_RAM'] -= request['SLICE_MEC_RAM_REQUEST']
            self.resources_1['MEC_STORAGE'] -= request['SLICE_MEC_STORAGE_REQUEST']
            self.resources_1['MEC_BW'] -= request['SLICE_MEC_BW_REQUEST']
        elif slice_id == 2:
            self.resources_2['MEC_CPU'] -= request['SLICE_MEC_CPU_REQUEST']
            self.resources_2['MEC_RAM'] -= request['SLICE_MEC_RAM_REQUEST']
            self.resources_2['MEC_STORAGE'] -= request['SLICE_MEC_STORAGE_REQUEST']
            self.resources_2['MEC_BW'] -= request['SLICE_MEC_BW_REQUEST']
        elif slice_id == 3:
            self.resources_3['MEC_CPU'] -= request['SLICE_MEC_CPU_REQUEST']
            self.resources_3['MEC_RAM'] -= request['SLICE_MEC_RAM_REQUEST']
            self.resources_3['MEC_STORAGE'] -= request['SLICE_MEC_STORAGE_REQUEST']
            self.resources_3['MEC_BW'] -= request['SLICE_MEC_BW_REQUEST']
        elif slice_id == 4:
            self.resources_4['MEC_CPU'] -= request['SLICE_MEC_CPU_REQUEST']
            self.resources_4['MEC_RAM'] -= request['SLICE_MEC_RAM_REQUEST']
            self.resources_4['MEC_STORAGE'] -= request['SLICE_MEC_STORAGE_REQUEST']
            self.resources_4['MEC_BW'] -= request['SLICE_MEC_BW_REQUEST']
        elif slice_id == 5:
            self.resources_5['MEC_CPU'] -= request['SLICE_MEC_CPU_REQUEST']
            self.resources_5['MEC_RAM'] -= request['SLICE_MEC_RAM_REQUEST']
            self.resources_5['MEC_STORAGE'] -= request['SLICE_MEC_STORAGE_REQUEST']
            self.resources_5['MEC_BW'] -= request['SLICE_MEC_BW_REQUEST']
        elif slice_id == 6:
            self.resources_6['MEC_CPU'] -= request['SLICE_MEC_CPU_REQUEST']
            self.resources_6['MEC_RAM'] -= request['SLICE_MEC_RAM_REQUEST']
            self.resources_6['MEC_STORAGE'] -= request['SLICE_MEC_STORAGE_REQUEST']
            self.resources_6['MEC_BW'] -= request['SLICE_MEC_BW_REQUEST']
    
    def deallocate_slice(self, request, slice_id):
        # Function to deallocate resources of killed requests

        # RAN Resources Part-----------------------------------------------------------
        indices = np.where(self.PRB_map == request['UE_ID'])

        for i in range(len(indices[0])):
            #print(f"({indices[0][i]}, {indices[1][i]})")
            self.PRB_map[indices[0][i], indices[1][i]] = 0

        # MEC Resources Part------------------------------------------------------------
        if slice_id == 1:
            self.resources_1['MEC_CPU'] += request['SLICE_MEC_CPU_REQUEST']
            self.resources_1['MEC_RAM'] += request['SLICE_MEC_RAM_REQUEST']
            self.resources_1['MEC_STORAGE'] += request['SLICE_MEC_STORAGE_REQUEST']
            self.resources_1['MEC_BW'] += request['SLICE_MEC_BW_REQUEST']
        elif slice_id == 2:
            self.resources_2['MEC_CPU'] += request['SLICE_MEC_CPU_REQUEST']
            self.resources_2['MEC_RAM'] += request['SLICE_MEC_RAM_REQUEST']
            self.resources_2['MEC_STORAGE'] += request['SLICE_MEC_STORAGE_REQUEST']
            self.resources_2['MEC_BW'] += request['SLICE_MEC_BW_REQUEST']
        elif slice_id == 3:
            self.resources_3['MEC_CPU'] += request['SLICE_MEC_CPU_REQUEST']
            self.resources_3['MEC_RAM'] += request['SLICE_MEC_RAM_REQUEST']
            self.resources_3['MEC_STORAGE'] += request['SLICE_MEC_STORAGE_REQUEST']
            self.resources_3['MEC_BW'] += request['SLICE_MEC_BW_REQUEST']
        elif slice_id == 4:
            self.resources_4['MEC_CPU'] += request['SLICE_MEC_CPU_REQUEST']
            self.resources_4['MEC_RAM'] += request['SLICE_MEC_RAM_REQUEST']
            self.resources_4['MEC_STORAGE'] += request['SLICE_MEC_STORAGE_REQUEST']
            self.resources_4['MEC_BW'] += request['SLICE_MEC_BW_REQUEST']
        elif slice_id == 5:
            self.resources_5['MEC_CPU'] += request['SLICE_MEC_CPU_REQUEST']
            self.resources_5['MEC_RAM'] += request['SLICE_MEC_RAM_REQUEST']
            self.resources_5['MEC_STORAGE'] += request['SLICE_MEC_STORAGE_REQUEST']
            self.resources_5['MEC_BW'] += request['SLICE_MEC_BW_REQUEST']
        elif slice_id == 6:
            self.resources_6['MEC_CPU'] += request['SLICE_MEC_CPU_REQUEST']
            self.resources_6['MEC_RAM'] += request['SLICE_MEC_RAM_REQUEST']
            self.resources_6['MEC_STORAGE'] += request['SLICE_MEC_STORAGE_REQUEST']
            self.resources_6['MEC_BW'] += request['SLICE_MEC_BW_REQUEST']
        
    def create_slice (self, request):
        # Function to create the slice for a specific request
        # This function inserts the defined slice to the request in the processed requests list
        #         self.slices_param = {1: [4, 16, 100, 40, 50], 2: [4, 32, 100, 100, 30], 3: [8, 16, 32, 80, 20], 
        #                               4: [4, 8, 16, 50, 25], 5: [2, 8, 32, 40, 10], 6: [2, 8, 32, 40, 5]}
        slice1 = self.slices_param[1]
        slice2 = self.slices_param[2]
        slice3 = self.slices_param[3]
        slice4 = self.slices_param[4]
        slice5 = self.slices_param[5]
        slice6 = self.slices_param[6]
        
        if (request['SLICE_RAN_R_REQUEST'] >= slice1[4]):
            slice_id = 1
        elif (request['SLICE_RAN_R_REQUEST'] >= slice2[4]):
            slice_id = 2
        elif (request['SLICE_RAN_R_REQUEST'] >= slice4[4]):
            slice_id = 4
        elif (request['SLICE_RAN_R_REQUEST'] >= slice3[4]):
            slice_id = 3
        elif (request['SLICE_RAN_R_REQUEST'] >= slice5[4]):
            slice_id = 5
        elif (request['SLICE_RAN_R_REQUEST'] >= slice6[4]):
            slice_id = 6
        return slice_id

    def reset_resources(self):
        #self.resources_1 = {'MEC_CPU': 30, 'MEC_RAM': 128, 'MEC_STORAGE': 100, 'MEC_BW': 300}
        #self.resources_2 = {'MEC_CPU': 30, 'MEC_RAM': 128, 'MEC_STORAGE': 100, 'MEC_BW': 200}
        #self.resources_3 = {'MEC_CPU': 50, 'MEC_RAM': 128, 'MEC_STORAGE': 100, 'MEC_BW': 200}
        #self.resources_4 = {'MEC_CPU': 30, 'MEC_RAM': 128, 'MEC_STORAGE': 100, 'MEC_BW': 200}
        #self.resources_5 = {'MEC_CPU': 20, 'MEC_RAM': 64, 'MEC_STORAGE': 80, 'MEC_BW': 100}
        #self.resources_6 = {'MEC_CPU': 20, 'MEC_RAM': 64, 'MEC_STORAGE': 80, 'MEC_BW': 80}

        self.PRB_map = np.zeros((14, self.PRB_per_channel))

        self.resources_1['MEC_CPU'] = 30
        self.resources_1['MEC_RAM'] = 128
        self.resources_1['MEC_STORAGE'] = 100
        self.resources_1['MEC_BW'] = 300

        self.resources_2['MEC_CPU'] = 30
        self.resources_2['MEC_RAM'] = 128
        self.resources_2['MEC_STORAGE'] = 100
        self.resources_2['MEC_BW'] = 200

        self.resources_3['MEC_CPU'] = 50
        self.resources_3['MEC_RAM'] = 128
        self.resources_3['MEC_STORAGE'] = 100
        self.resources_3['MEC_BW'] = 200

        self.resources_4['MEC_CPU'] = 30
        self.resources_4['MEC_RAM'] = 128
        self.resources_4['MEC_STORAGE'] = 100
        self.resources_4['MEC_BW'] = 200

        self.resources_5['MEC_CPU'] = 20
        self.resources_5['MEC_RAM'] = 64
        self.resources_5['MEC_STORAGE'] = 80
        self.resources_5['MEC_BW'] = 100

        self.resources_6['MEC_CPU'] = 20
        self.resources_6['MEC_RAM'] = 64
        self.resources_6['MEC_STORAGE'] = 80
        self.resources_6['MEC_BW'] = 80
    
    def evaluate_action(self, action, slice_id, reward_value, terminated):
        if action == 1 and slice_id == 1:
            self.check_resources(self.next_request)
            if self.resources_flag == 1:
                self.allocate_slice(self.next_request,slice_id)
                self.processed_requests[len(self.processed_requests) - 1]['SliceID'] = slice_id
                self.reward += reward_value   
                self.next_request = self.read_request()
            else: 
                terminated = True
                self.reward = 0
        
        if action == 1 and slice_id != 1:
            terminated = True
            self.reward = 0
            
        if action == 2 and slice_id == 2:
            self.check_resources(self.next_request)
            if self.resources_flag == 1:
                self.allocate_slice(self.next_request, slice_id)
                self.processed_requests[len(self.processed_requests) - 1]['SliceID'] = slice_id
                self.reward += reward_value   
                self.next_request = self.read_request()
            else: 
                terminated = True
                self.reward = 0
        
        if action == 2 and slice_id != 2:
            terminated = True
            self.reward = 0
            
        if action == 3 and slice_id == 3:
            self.check_resources(self.next_request)
            if self.resources_flag == 1:
                self.allocate_slice(self.next_request, slice_id)
                self.processed_requests[len(self.processed_requests) - 1]['SliceID'] = slice_id
                self.reward += reward_value   
                self.next_request = self.read_request()
            else: 
                terminated = True
                self.reward = 0
        
        if action == 3 and slice_id != 3:
            terminated = True
            self.reward = 0

        if action == 4 and slice_id == 4:
            self.check_resources(self.next_request)
            if self.resources_flag == 1:
                self.allocate_slice(self.next_request, slice_id)
                self.processed_requests[len(self.processed_requests) - 1]['SliceID'] = slice_id
                self.reward += reward_value   
                self.next_request = self.read_request()
            else: 
                terminated = True
                self.reward = 0
        
        if action == 4 and slice_id != 4:
            terminated = True
            self.reward = 0
            
        if action == 5 and slice_id == 5:
            self.check_resources(self.next_request)
            if self.resources_flag == 1:
                self.allocate_slice(self.next_request, slice_id)
                self.processed_requests[len(self.processed_requests) - 1]['SliceID'] = slice_id
                self.reward += reward_value   
                self.next_request = self.read_request()
            else: 
                terminated = True
                self.reward = 0
        
        if action == 5 and slice_id != 5:
            terminated = True
            self.reward = 0

        if action == 6 and slice_id == 6:
            self.check_resources(self.next_request)
            if self.resources_flag == 1:
                self.allocate_slice(self.next_request, slice_id)
                self.processed_requests[len(self.processed_requests) - 1]['SliceID'] = slice_id
                self.reward += reward_value   
                self.next_request = self.read_request()
            else: 
                terminated = True
                self.reward = 0
        
        if action == 6 and slice_id != 6:
            terminated = True
            self.reward = 0

        if action == 0:
            self.check_resources(self.next_request)
            if self.resources_flag == 0:
                self.reward += reward_value
                self.next_request = self.read_request()
            else: 
                terminated = True
                self.reward = 0        
        
        return terminated

    def check_RAN(self, request):
        indices = np.where(self.PRB_map == 0)
        available_symbols = len(indices[0])

        W_total = self.PRB_BW * self.sprectral_efficiency * available_symbols

        if request['SLICE_RAN_R_REQUEST'] * (10**6) <= W_total * log2(1 + request['UE_SiNR']):
            return True
        else: return False

    def allocate_ran(self, request):
        indices = np.where(self.PRB_map == 0)

        number_symbols = ceil((request['SLICE_RAN_R_REQUEST'] * (10**6)) / (self.PRB_BW * self.sprectral_efficiency * log2(1 + request['UE_SiNR'])))

        for i in range(number_symbols):
            #print(f"({indices[0][i]}, {indices[1][i]})")
            self.PRB_map[indices[0][i], indices[1][i]] = request['UE_ID']

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            
#a = SliceManagementEnv1()
#check_env(a)