import numpy as np
import random
import math
import queue

class Offload:
    def __init__(self, num_iot, num_fog, num_time, max_delay, task_arrive_prob):
        # --- Config ---
        self.n_iot = num_iot
        self.n_fog = num_fog
        self.n_time = num_time
        self.max_delay = max_delay
        self.duration = 0.3
        self.task_arrive_prob = task_arrive_prob
        self.min_bit_arrive = 0.5
        self.max_bit_arrive = 5.0

        # --- Reward Weights ---
        self.w1 = 0.5
        self.w2 = 0.5
        self.drop_penalty = 2.0 * self.max_delay

        # --- Physics ---
        self.height = 100
        self.ground_length = 100
        self.ground_width = 100
        self.bandwidth_nums = 2
        self.B = self.bandwidth_nums * 10 ** 6
        self.p_noisy_los = 10 ** (-13)
        self.p_noisy_nlos = 10 ** (-11)
        self.alpha0 = 1e-5
        self.p_uplink = 0.1
        self.t_move = 0.02
        self.v_ue = 1

        # --- Capabilities ---
        self.comp_cap_iot = 1.5 * np.ones(self.n_iot) * self.duration
        self.comp_cap_fog = 2.5 * np.ones(self.n_fog) * self.duration
        self.comp_cap_fog[0] = 5 * self.duration
        self.tran_cap_sat = 14 * self.duration
        self.propagation_sat = 0.2
        self.p_fog = 0.1 * np.ones(self.n_fog)
        self.p_fog[0] = 0.3
        self.coeff = 10 ** -27
        self.comp_density = 0.297 * np.ones(self.n_iot)

        # --- RL State ---
        self.n_actions = 1 + num_fog
        self.n_features = 2 + (2 * self.n_fog) + 1 + 1 + 1 + num_fog
        self.n_lstm_state = self.n_fog

        # --- Init Internal State ---
        self.time_count = 0
        self.bitArrive = np.zeros([self.n_time, self.n_iot])
        
        self.Queue_iot_comp = [queue.Queue() for _ in range(self.n_iot)]
        self.Queue_iot_tran = [queue.Queue() for _ in range(self.n_iot)]
        self.Queue_fog_comp = [[queue.Queue() for _ in range(self.n_fog)] for _ in range(self.n_iot)]

        self.t_iot_comp = -np.ones(self.n_iot)
        self.t_iot_tran = -np.ones(self.n_iot)
        self.b_fog_comp = np.zeros((self.n_iot, self.n_fog))
        self.fog_iot_m = np.zeros(self.n_fog)
        self.fog_iot_m_observe = np.zeros(self.n_fog)

        self.task_on_process_local = [{'size': np.nan, 'time': np.nan, 'remain': np.nan} for _ in range(self.n_iot)]
        self.task_on_transmit_local = [{'size': np.nan, 'time': np.nan, 'fog': np.nan, 'remain': np.nan} for _ in range(self.n_iot)]
        self.task_on_process_fog = [[{'size': np.nan, 'time': np.nan, 'remain': np.nan, 'energy': np.nan} for _ in range(self.n_fog)] for _ in range(self.n_iot)]

        self.process_delay = np.zeros([self.n_time, self.n_iot])
        self.process_delay_unfinish_ind = np.zeros([self.n_time, self.n_iot])
        self.process_delay_trans = np.zeros([self.n_time, self.n_iot])
        self.process_energy = np.zeros([self.n_time, self.n_iot])
        self.process_energy_trans = np.zeros([self.n_time, self.n_iot])
        self.fog_drop = np.zeros([self.n_iot, self.n_fog])

        self.drop_trans_count = 0
        self.drop_fog_count = 0
        self.drop_iot_count = 0

        self.loc_ue_list = np.random.randint(0, 101, size=2 * self.n_iot)
        self.loc_uav_list = np.random.randint(0, 101, size=2 * self.n_fog)

    def reset(self, bitArrive):
        self.bitArrive = bitArrive
        self.time_count = 0
        self.loc_ue_list = np.random.randint(0, 101, size=2 * self.n_iot)
        
        # Clear Queues
        self.Queue_iot_comp = [queue.Queue() for _ in range(self.n_iot)]
        self.Queue_iot_tran = [queue.Queue() for _ in range(self.n_iot)]
        self.Queue_fog_comp = [[queue.Queue() for _ in range(self.n_fog)] for _ in range(self.n_iot)]

        # Clear Trackers
        self.t_iot_comp.fill(-1)
        self.t_iot_tran.fill(-1)
        self.b_fog_comp.fill(0)
        self.fog_iot_m.fill(0)
        self.fog_iot_m_observe.fill(0)
        self.fog_drop.fill(0)

        # Clear Active Tasks
        for i in range(self.n_iot):
            self.task_on_process_local[i] = {'size': np.nan, 'time': np.nan, 'remain': np.nan}
            self.task_on_transmit_local[i] = {'size': np.nan, 'time': np.nan, 'fog': np.nan, 'remain': np.nan}
            for f in range(self.n_fog):
                self.task_on_process_fog[i][f] = {'size': np.nan, 'time': np.nan, 'remain': np.nan, 'energy': np.nan}

        # Clear logs
        self.process_delay.fill(0)
        self.process_energy.fill(0)
        self.process_delay_unfinish_ind.fill(0)
        self.drop_trans_count = 0
        self.drop_fog_count = 0
        self.drop_iot_count = 0

        return self._get_observation(), np.zeros((self.n_iot, self.n_lstm_state))

    def _calculate_reward(self, delay, energy, is_dropped):
        if is_dropped:
            return -self.drop_penalty
        return -(self.w1 * delay + self.w2 * energy)

    def calc_tran(self, fog_cur, iot_ind):
        dx = self.loc_uav_list[2 * fog_cur] - self.loc_ue_list[2 * iot_ind]
        dy = self.loc_uav_list[2 * fog_cur + 1] - self.loc_ue_list[2 * iot_ind + 1]
        dist = np.sqrt(dx**2 + dy**2 + self.height**2)
        g_uav_ue = abs(self.alpha0 / dist**2)
        return self.duration * self.B * math.log2(1 + self.p_uplink * g_uav_ue / self.p_noisy_los) * 1e-6

    def step(self, action):
        finished_tasks = []

        # 1. Action Parsing
        iot_action_local = np.zeros(self.n_iot, dtype=int)
        iot_action_fog = np.zeros(self.n_iot, dtype=int)
        for i in range(self.n_iot):
            a = action[i]
            iot_action_fog[i] = int(a - 1)
            if a == 0:
                iot_action_local[i] = 1

        # 2. Local Computation
        for i in range(self.n_iot):
            if self.bitArrive[self.time_count, i] > 0 and iot_action_local[i] == 1:
                self.Queue_iot_comp[i].put({'size': self.bitArrive[self.time_count, i], 'time': self.time_count})

            if math.isnan(self.task_on_process_local[i]['remain']) and not self.Queue_iot_comp[i].empty():
                while not self.Queue_iot_comp[i].empty():
                    task = self.Queue_iot_comp[i].get()
                    if task['size'] == 0: continue
                    
                    if self.time_count - task['time'] + 1 > self.max_delay:
                        # Drop
                        self.process_delay[task['time'], i] = self.max_delay
                        self.process_delay_unfinish_ind[task['time'], i] = 1
                        r = self._calculate_reward(self.max_delay, 0, True)
                        finished_tasks.append({'iot': i, 'start_time': task['time'], 'reward': r, 'dropped': True, 'delay': self.max_delay, 'energy': 0.0})
                        self.drop_iot_count += 1
                    else:
                        self.task_on_process_local[i] = {'size': task['size'], 'time': task['time'], 'remain': task['size']}
                        break

            if not math.isnan(self.task_on_process_local[i]['remain']):
                capacity = self.comp_cap_iot[i] / self.comp_density[i]
                self.task_on_process_local[i]['remain'] -= capacity
                
                if self.task_on_process_local[i]['remain'] <= 0:
                    delay = self.time_count - self.task_on_process_local[i]['time'] + 1
                    energy = self.coeff * (((self.comp_cap_iot[i]/self.duration)*1e9)**2) * (self.comp_density[i]*1e9) * self.task_on_process_local[i]['size'] * 1e-2
                    
                    # FIX 2: Logging restored
                    self.process_delay[self.task_on_process_local[i]['time'], i] = delay
                    self.process_energy[self.task_on_process_local[i]['time'], i] = energy
                    
                    r = self._calculate_reward(delay, energy, False)
                    finished_tasks.append({'iot': i, 'start_time': self.task_on_process_local[i]['time'], 'reward': r, 'dropped': False, 'delay': delay, 'energy': energy})
                    self.task_on_process_local[i]['remain'] = np.nan
                elif self.time_count - self.task_on_process_local[i]['time'] + 1 >= self.max_delay:
                    # Timeout
                    self.process_delay[self.task_on_process_local[i]['time'], i] = self.max_delay
                    self.process_delay_unfinish_ind[self.task_on_process_local[i]['time'], i] = 1
                    r = self._calculate_reward(self.max_delay, 0, True)
                    finished_tasks.append({'iot': i, 'start_time': self.task_on_process_local[i]['time'], 'reward': r, 'dropped': True, 'delay': self.max_delay, 'energy': 0.0})
                    self.task_on_process_local[i]['remain'] = np.nan
                    self.drop_iot_count += 1

            # Update Wait Time
            if self.bitArrive[self.time_count, i] > 0:
                est_wait = self.t_iot_comp[i] + 1
                if self.t_iot_comp[i] < self.time_count: est_wait = self.time_count
                if iot_action_local[i] == 1:
                    est_wait += math.ceil(self.bitArrive[self.time_count, i] / (self.comp_cap_iot[i] / self.comp_density[i]))
                self.t_iot_comp[i] = min(est_wait, self.time_count + self.max_delay)

        # 3. Transmission
        for i in range(self.n_iot):
            if self.bitArrive[self.time_count, i] > 0 and iot_action_local[i] == 0:
                self.Queue_iot_tran[i].put({'size': self.bitArrive[self.time_count, i], 'time': self.time_count, 'fog': iot_action_fog[i]})

            if math.isnan(self.task_on_transmit_local[i]['remain']) and not self.Queue_iot_tran[i].empty():
                while not self.Queue_iot_tran[i].empty():
                    task = self.Queue_iot_tran[i].get()
                    if task['size'] == 0: continue

                    if self.time_count - task['time'] + 1 > self.max_delay:
                        self.process_delay[task['time'], i] = self.max_delay
                        self.process_delay_unfinish_ind[task['time'], i] = 1
                        r = self._calculate_reward(self.max_delay, 0, True)
                        finished_tasks.append({'iot': i, 'start_time': task['time'], 'reward': r, 'dropped': True, 'delay': self.max_delay, 'energy': 0.0})
                        self.drop_trans_count += 1
                    else:
                        self.task_on_transmit_local[i] = {'size': task['size'], 'time': task['time'], 'fog': task['fog'], 'remain': task['size']}
                        break

            if not math.isnan(self.task_on_transmit_local[i]['remain']):
                f = int(self.task_on_transmit_local[i]['fog'])
                rate = self.tran_cap_sat if f == 0 else self.calc_tran(f, i)
                self.task_on_transmit_local[i]['remain'] -= rate

                if self.task_on_transmit_local[i]['remain'] <= 0:
                    e_trans = 0
                    if f == 0:
                        e_trans = self.p_fog[0] * self.task_on_transmit_local[i]['size'] / (self.tran_cap_sat/self.duration)
                    else:
                        e_trans = self.p_fog[f] * self.task_on_transmit_local[i]['size'] / (self.calc_tran(f, i)/self.duration)
                    
                    self.Queue_fog_comp[i][f].put({
                        'size': self.task_on_transmit_local[i]['size'],
                        'time': self.task_on_transmit_local[i]['time'],
                        'energy_trans': e_trans
                    })
                    self.b_fog_comp[i][f] += self.task_on_transmit_local[i]['size']
                    self.task_on_transmit_local[i]['remain'] = np.nan
                
                elif self.time_count - self.task_on_transmit_local[i]['time'] + 1 >= self.max_delay:
                    self.process_delay[self.task_on_transmit_local[i]['time'], i] = self.max_delay
                    self.process_delay_unfinish_ind[self.task_on_transmit_local[i]['time'], i] = 1
                    r = self._calculate_reward(self.max_delay, 0, True)
                    finished_tasks.append({'iot': i, 'start_time': self.task_on_transmit_local[i]['time'], 'reward': r, 'dropped': True, 'delay': self.max_delay, 'energy': 0.0})
                    self.task_on_transmit_local[i]['remain'] = np.nan
                    self.drop_trans_count += 1

            if self.bitArrive[self.time_count, i] > 0:
                est_wait = self.t_iot_tran[i] + 1
                if self.t_iot_tran[i] < self.time_count: est_wait = self.time_count
                if iot_action_local[i] == 0:
                    f = iot_action_fog[i]
                    rate = self.tran_cap_sat if f == 0 else self.calc_tran(f, i)
                    add_t = math.ceil(self.bitArrive[self.time_count, i] / rate)
                    if f == 0: add_t += 2 * self.propagation_sat
                    est_wait += add_t
                self.t_iot_tran[i] = min(est_wait, self.time_count + self.max_delay)

        # 4. Fog Computation
        for i in range(self.n_iot):
            for f in range(self.n_fog):
                if math.isnan(self.task_on_process_fog[i][f]['remain']) and not self.Queue_fog_comp[i][f].empty():
                    while not self.Queue_fog_comp[i][f].empty():
                        task = self.Queue_fog_comp[i][f].get()
                        if self.time_count - task['time'] + 1 > self.max_delay:
                            self.process_delay[task['time'], i] = self.max_delay
                            self.process_delay_unfinish_ind[task['time'], i] = 1
                            r = self._calculate_reward(self.max_delay, 0, True)
                            finished_tasks.append({'iot': i, 'start_time': task['time'], 'reward': r, 'dropped': True, 'delay': self.max_delay, 'energy': 0.0})
                            self.drop_fog_count += 1
                            # FIX 1: Remove Ghost Load on Queue Drop
                            self.b_fog_comp[i][f] = max(0, self.b_fog_comp[i][f] - task['size'])
                        else:
                            self.task_on_process_fog[i][f] = {'size': task['size'], 'time': task['time'], 'remain': task['size'], 'energy': task['energy_trans']}
                            break

                if not math.isnan(self.task_on_process_fog[i][f]['remain']):
                    share = self.fog_iot_m[f] if self.fog_iot_m[f] > 0 else 1
                    capacity = self.comp_cap_fog[f] / self.comp_density[i] / share
                    self.task_on_process_fog[i][f]['remain'] -= capacity
                    self.b_fog_comp[i][f] = max(0, self.b_fog_comp[i][f] - capacity)

                    if self.task_on_process_fog[i][f]['remain'] <= 0:
                        delay = self.time_count - self.task_on_process_fog[i][f]['time'] + 1
                        comp_energy = self.coeff * (((self.comp_cap_fog[f]/self.duration)*1e9)**2) * (self.comp_density[i]*1e9) * self.task_on_process_fog[i][f]['size'] * 1e-2
                        total_energy = self.task_on_process_fog[i][f]['energy'] + comp_energy
                        
                        # FIX 2: Logging restored
                        self.process_delay[self.task_on_process_fog[i][f]['time'], i] = delay
                        self.process_energy[self.task_on_process_fog[i][f]['time'], i] = total_energy

                        r = self._calculate_reward(delay, total_energy, False)
                        finished_tasks.append({'iot': i, 'start_time': self.task_on_process_fog[i][f]['time'], 'reward': r, 'dropped': False, 'delay': delay, 'energy': total_energy})
                        self.task_on_process_fog[i][f]['remain'] = np.nan
                    
                    elif self.time_count - self.task_on_process_fog[i][f]['time'] + 1 >= self.max_delay:
                        self.process_delay[self.task_on_process_fog[i][f]['time'], i] = self.max_delay
                        self.process_delay_unfinish_ind[self.task_on_process_fog[i][f]['time'], i] = 1
                        
                        # FIX 1: Set Fog Drop before subtracting
                        self.fog_drop[i, f] = self.task_on_process_fog[i][f]['remain']
                        
                        r = self._calculate_reward(self.max_delay, 0, True)
                        finished_tasks.append({'iot': i, 'start_time': self.task_on_process_fog[i][f]['time'], 'reward': r, 'dropped': True, 'delay': self.max_delay, 'energy': 0.0})
                        
                        self.task_on_process_fog[i][f]['remain'] = np.nan
                        self.drop_fog_count += 1
                        self.b_fog_comp[i][f] = max(0, self.b_fog_comp[i][f] - self.fog_drop[i, f])

        # 5. Congestion Update (FIXED ORDER)
        self.fog_iot_m.fill(0)
        for f in range(self.n_fog):
            for i in range(self.n_iot):
                if (not self.Queue_fog_comp[i][f].empty()
                    or not math.isnan(self.task_on_process_fog[i][f]['remain'])
                    or self.b_fog_comp[i][f] > 0):
                    self.fog_iot_m[f] += 1

        # Observe CURRENT congestion
        self.fog_iot_m_observe = self.fog_iot_m.copy()


        # 6. Update
        self.time_count += 1
        done = (self.time_count >= self.n_time)

        if done:
            finished_tasks.extend(self._finalize_episode())

        # Move
        for i in range(self.n_iot):
            tmp = np.random.rand(2)
            theta = tmp[0] * np.pi * 2
            dist = tmp[1] * self.t_move * self.v_ue
            self.loc_ue_list[2*i] = np.clip(self.loc_ue_list[2*i] + math.cos(theta)*dist, 0, self.ground_length)
            self.loc_ue_list[2*i+1] = np.clip(self.loc_ue_list[2*i+1] + math.sin(theta)*dist, 0, self.ground_width)

        obs = self._get_observation()
        lstm_state = np.tile(self.fog_iot_m_observe, (self.n_iot, 1))

        return obs, lstm_state, done, {'finished': finished_tasks}

    def _finalize_episode(self):
        penalties = []
        r_drop = self._calculate_reward(self.max_delay, 0, True)

        for i in range(self.n_iot):
            q_list = [self.Queue_iot_comp[i], self.Queue_iot_tran[i]] + [self.Queue_fog_comp[i][f] for f in range(self.n_fog)]
            for q in q_list:
                while not q.empty():
                    t = q.get()
                    if t['size'] > 0:
                        self.process_delay[t['time'], i] = self.max_delay
                        self.process_delay_unfinish_ind[t['time'], i] = 1
                        penalties.append({'iot': i, 'start_time': t['time'], 'reward': r_drop, 'dropped': True, 'delay': self.max_delay, 'energy': 0.0})

            if not math.isnan(self.task_on_process_local[i]['remain']):
                t = self.task_on_process_local[i]['time']
                self.process_delay[t, i] = self.max_delay
                self.process_delay_unfinish_ind[t, i] = 1
                penalties.append({'iot': i, 'start_time': t, 'reward': r_drop, 'dropped': True, 'delay': self.max_delay, 'energy': 0.0})
            
            if not math.isnan(self.task_on_transmit_local[i]['remain']):
                t = self.task_on_transmit_local[i]['time']
                self.process_delay[t, i] = self.max_delay
                self.process_delay_unfinish_ind[t, i] = 1
                penalties.append({'iot': i, 'start_time': t, 'reward': r_drop, 'dropped': True, 'delay': self.max_delay, 'energy': 0.0})

            for f in range(self.n_fog):
                if not math.isnan(self.task_on_process_fog[i][f]['remain']):
                    t = self.task_on_process_fog[i][f]['time']
                    self.process_delay[t, i] = self.max_delay
                    self.process_delay_unfinish_ind[t, i] = 1
                    penalties.append({'iot': i, 'start_time': t, 'reward': r_drop, 'dropped': True, 'delay': self.max_delay, 'energy': 0.0})
        
        return penalties

    def _get_observation(self):
        obs = np.zeros((self.n_iot, self.n_features))
        if self.time_count >= self.n_time: return obs
        
        for i in range(self.n_iot):
            # FIX: Always populate state
            obs[i, :] = np.hstack([
                self.loc_ue_list[2*i:2*i+2],
                self.loc_uav_list,
                self.bitArrive[self.time_count, i], 
                self.t_iot_comp[i] - self.time_count + 1,
                self.t_iot_tran[i] - self.time_count + 1,
                self.b_fog_comp[i, :].flatten()
            ])
        return obs
    
    def get_global_state(self):
        return np.concatenate([
        np.array(self.task_on_process_fog, dtype=np.float32),   # fog loads
        np.array(self.Queue_fog_comp, dtype=np.float32),        # fog queue lengths
        np.array(self.p_fog, dtype=np.float32),                 # channel states
        np.array([self.time_count], dtype=np.float32)           # global time
    ], axis=0)
