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
        
        # ---- Energy nonlinearity parameters ----
        self.alpha_comp = 2.5      # computation convexity
        self.beta_tx = 1.8         # transmission convexity

        self.kappa_comp = 1.0      # scaling constants
        self.kappa_tx = 1.0

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
        self.soft_drop_count = 0
        self.episode_soft_drops = 0
        self.episode_hard_drops = 0
        self.episode_success_count = 0
        self.episode_energy_sum = 0.0

        self.loc_ue_list = np.random.randint(0, 101, size=2 * self.n_iot)
        self.loc_uav_list = np.random.randint(0, 101, size=2 * self.n_fog)
        
        self.soft_drop_count = 0
        # Spillover activates if neighbor has too many active tasks

        self.log_energy = []

        # ---- Episode return ----
        self.episode_return = 0.0
        self.log_episode_return = []

        # ---- Fog utilization ----
        self.log_fog_var = []          # per episode
        self._fog_var_accumulator = [] # per timestep

        self.log_fog_delta = []        # Lyapunov proxy
        self._prev_fog_load = None

        # ---- Drop rate ----
        self.log_drop_rate = []

        # ---- Policy churn ----
        self.prev_actions = None
        self.log_policy_churn = []

        # ---- Stability marker ----
        self.episodes_to_stability = None
        self.total_tasks = 0

        # ===============================
        # Bursty arrival process (MMOOP)
        # ===============================

        # Markov transition probabilities
        self.burst_p_on  = 0.02   # OFF → ON
        self.burst_p_off = 0.07   # ON → OFF

        # Arrival probabilities
        self.burst_prob_on  = 0.93
        self.burst_prob_off = 0.12   # tuned so avg λ ≈ 0.25

        # Burst state per IoT (0 = OFF, 1 = ON)
        self.iot_burst_state = np.zeros(self.n_iot, dtype=np.int8)


    def reset(self, bitArrive):
        self.bitArrive = bitArrive
        self.time_count = 0
        # Reset burst state
        self.iot_burst_state[:] = 0
        self.loc_ue_list = np.random.randint(0, 101, size=2 * self.n_iot)
        self.prev_actions = None
        self.soft_drop_count = 0
        self.episode_return = 0.0
        self.episode_soft_drops = 0
        self.episode_hard_drops = 0
        self.episode_success_count = 0
        self.episode_energy_sum = 0.0
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
        self.soft_drop_count = 0
        self._fog_var_accumulator.clear()
        self.episode_energy_sum = 0.0
        self.total_tasks = 0

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
    
    def _nonlinear_comp_energy(self, load, base_energy):
        """
        Energy = baseline (idle) + convex marginal cost.
        Captures amortization at moderate load and penalty at high load.
        """
        load = np.clip(load, 0.0, 1.0)

        # ---- NEW: idle / baseline energy ----
        E_idle = 0.3 * base_energy   # 20–40% is realistic

        # ---- Existing convex marginal term ----
        E_marginal = self.kappa_comp * (load ** self.alpha_comp) * base_energy

        return E_idle + E_marginal

    # def step(self, action):
    #     finished_tasks = []

    #     # 1. Action Parsing
    #     iot_action_local = np.zeros(self.n_iot, dtype=int)
    #     iot_action_fog = np.zeros(self.n_iot, dtype=int)
    #     for i in range(self.n_iot):
    #         a = action[i]
    #         iot_action_fog[i] = int(a - 1)
    #         if a == 0:
    #             iot_action_local[i] = 1

    #     # 2. Local Computation
    #     for i in range(self.n_iot):
    #         if self.bitArrive[self.time_count, i] > 0 and iot_action_local[i] == 1:
    #             self.Queue_iot_comp[i].put({'size': self.bitArrive[self.time_count, i], 'time': self.time_count})

    #         if math.isnan(self.task_on_process_local[i]['remain']) and not self.Queue_iot_comp[i].empty():
    #             while not self.Queue_iot_comp[i].empty():
    #                 task = self.Queue_iot_comp[i].get()
    #                 if task['size'] == 0: continue
                    
    #                 if self.time_count - task['time'] + 1 > self.max_delay:
    #                     # Drop
    #                     self.process_delay[task['time'], i] = self.max_delay
    #                     self.process_delay_unfinish_ind[task['time'], i] = 1
    #                     r = self._calculate_reward(self.max_delay, 0, True)
    #                     finished_tasks.append({'iot': i, 'start_time': task['time'], 'reward': r, 'dropped': True, 'delay': self.max_delay, 'energy': 0.0})
    #                     self.drop_iot_count += 1
    #                 else:
    #                     self.task_on_process_local[i] = {'size': task['size'], 'time': task['time'], 'remain': task['size']}
    #                     break

    #         if not math.isnan(self.task_on_process_local[i]['remain']):
    #             capacity = self.comp_cap_iot[i] / self.comp_density[i]
    #             self.task_on_process_local[i]['remain'] -= capacity
                
    #             if self.task_on_process_local[i]['remain'] <= 0:
    #                 delay = self.time_count - self.task_on_process_local[i]['time'] + 1
    #                 energy = self.coeff * (((self.comp_cap_iot[i]/self.duration)*1e9)**2) * (self.comp_density[i]*1e9) * self.task_on_process_local[i]['size'] * 1e-2
                    
    #                 # FIX 2: Logging restored
    #                 self.process_delay[self.task_on_process_local[i]['time'], i] = delay
    #                 self.process_energy[self.task_on_process_local[i]['time'], i] = energy
                    
    #                 r = self._calculate_reward(delay, energy, False)
    #                 finished_tasks.append({'iot': i, 'start_time': self.task_on_process_local[i]['time'], 'reward': r, 'dropped': False, 'delay': delay, 'energy': energy})
    #                 self.task_on_process_local[i]['remain'] = np.nan
    #             elif self.time_count - self.task_on_process_local[i]['time'] + 1 >= self.max_delay:
    #                 # Timeout
    #                 self.process_delay[self.task_on_process_local[i]['time'], i] = self.max_delay
    #                 self.process_delay_unfinish_ind[self.task_on_process_local[i]['time'], i] = 1
    #                 r = self._calculate_reward(self.max_delay, 0, True)
    #                 finished_tasks.append({'iot': i, 'start_time': self.task_on_process_local[i]['time'], 'reward': r, 'dropped': True, 'delay': self.max_delay, 'energy': 0.0})
    #                 self.task_on_process_local[i]['remain'] = np.nan
    #                 self.drop_iot_count += 1

    #         # Update Wait Time
    #         if self.bitArrive[self.time_count, i] > 0:
    #             est_wait = self.t_iot_comp[i] + 1
    #             if self.t_iot_comp[i] < self.time_count: est_wait = self.time_count
    #             if iot_action_local[i] == 1:
    #                 est_wait += math.ceil(self.bitArrive[self.time_count, i] / (self.comp_cap_iot[i] / self.comp_density[i]))
    #             self.t_iot_comp[i] = min(est_wait, self.time_count + self.max_delay)

    #     # 3. Transmission
    #     for i in range(self.n_iot):
    #         if self.bitArrive[self.time_count, i] > 0 and iot_action_local[i] == 0:
    #             self.Queue_iot_tran[i].put({'size': self.bitArrive[self.time_count, i], 'time': self.time_count, 'fog': iot_action_fog[i]})

    #         if math.isnan(self.task_on_transmit_local[i]['remain']) and not self.Queue_iot_tran[i].empty():
    #             while not self.Queue_iot_tran[i].empty():
    #                 task = self.Queue_iot_tran[i].get()
    #                 if task['size'] == 0: continue

    #                 if self.time_count - task['time'] + 1 > self.max_delay:
    #                     self.process_delay[task['time'], i] = self.max_delay
    #                     self.process_delay_unfinish_ind[task['time'], i] = 1
    #                     r = self._calculate_reward(self.max_delay, 0, True)
    #                     finished_tasks.append({'iot': i, 'start_time': task['time'], 'reward': r, 'dropped': True, 'delay': self.max_delay, 'energy': 0.0})
    #                     self.drop_trans_count += 1
    #                 else:
    #                     self.task_on_transmit_local[i] = {'size': task['size'], 'time': task['time'], 'fog': task['fog'], 'remain': task['size']}
    #                     break

    #         if not math.isnan(self.task_on_transmit_local[i]['remain']):
    #             f = int(self.task_on_transmit_local[i]['fog'])
    #             rate = self.tran_cap_sat if f == 0 else self.calc_tran(f, i)
    #             self.task_on_transmit_local[i]['remain'] -= rate

    #             if self.task_on_transmit_local[i]['remain'] <= 0:
    #                 e_trans = 0
    #                 if f == 0:
    #                     e_trans = self.p_fog[0] * self.task_on_transmit_local[i]['size'] / (self.tran_cap_sat/self.duration)
    #                 else:
    #                     e_trans = self.p_fog[f] * self.task_on_transmit_local[i]['size'] / (self.calc_tran(f, i)/self.duration)
                    
    #                 self.Queue_fog_comp[i][f].put({
    #                     'size': self.task_on_transmit_local[i]['size'],
    #                     'time': self.task_on_transmit_local[i]['time'],
    #                     'energy_trans': e_trans
    #                 })
    #                 self.b_fog_comp[i][f] += self.task_on_transmit_local[i]['size']
    #                 self.task_on_transmit_local[i]['remain'] = np.nan
                
    #             elif self.time_count - self.task_on_transmit_local[i]['time'] + 1 >= self.max_delay:
    #                 self.process_delay[self.task_on_transmit_local[i]['time'], i] = self.max_delay
    #                 self.process_delay_unfinish_ind[self.task_on_transmit_local[i]['time'], i] = 1
    #                 r = self._calculate_reward(self.max_delay, 0, True)
    #                 finished_tasks.append({'iot': i, 'start_time': self.task_on_transmit_local[i]['time'], 'reward': r, 'dropped': True, 'delay': self.max_delay, 'energy': 0.0})
    #                 self.task_on_transmit_local[i]['remain'] = np.nan
    #                 self.drop_trans_count += 1

    #         if self.bitArrive[self.time_count, i] > 0:
    #             est_wait = self.t_iot_tran[i] + 1
    #             if self.t_iot_tran[i] < self.time_count: est_wait = self.time_count
    #             if iot_action_local[i] == 0:
    #                 f = iot_action_fog[i]
    #                 rate = self.tran_cap_sat if f == 0 else self.calc_tran(f, i)
    #                 add_t = math.ceil(self.bitArrive[self.time_count, i] / rate)
    #                 if f == 0: add_t += 2 * self.propagation_sat
    #                 est_wait += add_t
    #             self.t_iot_tran[i] = min(est_wait, self.time_count + self.max_delay)

    #     # 4. Fog Computation
    #     for i in range(self.n_iot):
    #         for f in range(self.n_fog):
    #             if math.isnan(self.task_on_process_fog[i][f]['remain']) and not self.Queue_fog_comp[i][f].empty():
    #                 while not self.Queue_fog_comp[i][f].empty():
    #                     task = self.Queue_fog_comp[i][f].get()
    #                     if self.time_count - task['time'] + 1 > self.max_delay:
    #                         self.process_delay[task['time'], i] = self.max_delay
    #                         self.process_delay_unfinish_ind[task['time'], i] = 1
    #                         r = self._calculate_reward(self.max_delay, 0, True)
    #                         finished_tasks.append({'iot': i, 'start_time': task['time'], 'reward': r, 'dropped': True, 'delay': self.max_delay, 'energy': 0.0})
    #                         self.drop_fog_count += 1
    #                         # FIX 1: Remove Ghost Load on Queue Drop
    #                         self.b_fog_comp[i][f] = max(0, self.b_fog_comp[i][f] - task['size'])
    #                     else:
    #                         self.task_on_process_fog[i][f] = {'size': task['size'], 'time': task['time'], 'remain': task['size'], 'energy': task['energy_trans']}
    #                         break

    #             if not math.isnan(self.task_on_process_fog[i][f]['remain']):
    #                 share = self.fog_iot_m[f] if self.fog_iot_m[f] > 0 else 1
    #                 capacity = self.comp_cap_fog[f] / self.comp_density[i] / share
    #                 self.task_on_process_fog[i][f]['remain'] -= capacity
    #                 self.b_fog_comp[i][f] = max(0, self.b_fog_comp[i][f] - capacity)

    #                 if self.task_on_process_fog[i][f]['remain'] <= 0:
    #                     delay = self.time_count - self.task_on_process_fog[i][f]['time'] + 1
    #                     comp_energy = self.coeff * (((self.comp_cap_fog[f]/self.duration)*1e9)**2) * (self.comp_density[i]*1e9) * self.task_on_process_fog[i][f]['size'] * 1e-2
    #                     total_energy = self.task_on_process_fog[i][f]['energy'] + comp_energy
                        
    #                     # FIX 2: Logging restored
    #                     self.process_delay[self.task_on_process_fog[i][f]['time'], i] = delay
    #                     self.process_energy[self.task_on_process_fog[i][f]['time'], i] = total_energy

    #                     r = self._calculate_reward(delay, total_energy, False)
    #                     finished_tasks.append({'iot': i, 'start_time': self.task_on_process_fog[i][f]['time'], 'reward': r, 'dropped': False, 'delay': delay, 'energy': total_energy})
    #                     self.task_on_process_fog[i][f]['remain'] = np.nan
                    
    #                 elif self.time_count - self.task_on_process_fog[i][f]['time'] + 1 >= self.max_delay:
    #                     self.process_delay[self.task_on_process_fog[i][f]['time'], i] = self.max_delay
    #                     self.process_delay_unfinish_ind[self.task_on_process_fog[i][f]['time'], i] = 1
                        
    #                     # FIX 1: Set Fog Drop before subtracting
    #                     self.fog_drop[i, f] = self.task_on_process_fog[i][f]['remain']
                        
    #                     r = self._calculate_reward(self.max_delay, 0, True)
    #                     finished_tasks.append({'iot': i, 'start_time': self.task_on_process_fog[i][f]['time'], 'reward': r, 'dropped': True, 'delay': self.max_delay, 'energy': 0.0})
                        
    #                     self.task_on_process_fog[i][f]['remain'] = np.nan
    #                     self.drop_fog_count += 1
    #                     self.b_fog_comp[i][f] = max(0, self.b_fog_comp[i][f] - self.fog_drop[i, f])

    #     # 5. Congestion Update (FIXED ORDER)
    #     self.fog_iot_m.fill(0)
    #     for f in range(self.n_fog):
    #         for i in range(self.n_iot):
    #             if (not self.Queue_fog_comp[i][f].empty()
    #                 or not math.isnan(self.task_on_process_fog[i][f]['remain'])
    #                 or self.b_fog_comp[i][f] > 0):
    #                 self.fog_iot_m[f] += 1

    #     # Observe CURRENT congestion
    #     self.fog_iot_m_observe = self.fog_iot_m.copy()


    #     # 6. Update
    #     self.time_count += 1
    #     done = (self.time_count >= self.n_time)

    #     if done:
    #         finished_tasks.extend(self._finalize_episode())

    #     # Move
    #     for i in range(self.n_iot):
    #         tmp = np.random.rand(2)
    #         theta = tmp[0] * np.pi * 2
    #         dist = tmp[1] * self.t_move * self.v_ue
    #         self.loc_ue_list[2*i] = np.clip(self.loc_ue_list[2*i] + math.cos(theta)*dist, 0, self.ground_length)
    #         self.loc_ue_list[2*i+1] = np.clip(self.loc_ue_list[2*i+1] + math.sin(theta)*dist, 0, self.ground_width)

    #     obs = self._get_observation()
    #     lstm_state = np.tile(self.fog_iot_m_observe, (self.n_iot, 1))

    #     return obs, lstm_state, done, {'finished': finished_tasks}
    # --- [NEW] Helper 1: The "Hockey Stick" Congestion ---
    def _get_congestion_penalty(self, fog_index):
        """
        Calculates processing slowdown based on node utilization.
        Standard M/M/1 approximation: Delay explodes as utilization -> 1.0
        """
        # Count active tasks on this node (Queue + Processing)
        active_tasks = self.fog_iot_m[fog_index]
        
        # Utilization: We assume the node 'chokes' as it approaches serving all IoTs at once
        # You can tune 'n_iot' to be a specific capacity limit if needed.
        utilization = active_tasks / self.n_iot
        
        # Parameters (The Trap)
        # k=4: Flat safe zone, then sudden spike
        # alpha=3.0: 4x slower at full load
        k_factor = 5.0
        alpha = 3.5
        
        penalty_factor = 1.0 + alpha * (utilization ** k_factor)
        return penalty_factor

    def _nonlinear_tx_energy(self, rate, max_rate, base_energy):
        """
        Convex transmission energy: inefficiency near saturation.
        """
        load = np.clip(rate / max_rate, 0.0, 1.0)

        E_idle = 0.2 * base_energy
        E_marginal = self.kappa_tx * (load ** self.beta_tx) * base_energy

        return E_idle + E_marginal


    # --- [NEW] Helper 2: The "Soft" Drop Probability ---
    def _check_soft_drop(self, current_delay):
        """
        Determines if a task drops based on probabilistic risk (Sigmoid).
        Replaces hard 'if delay > max_delay'.
        """
        sensitivity = 2.0  # How fuzzy is the deadline?
        
        # Sigmoid Probability
        # If delay == max, prob is 50%. 
        # If delay >> max, prob -> 100%
        delay_diff = current_delay - self.max_delay
        prob = 1.0 / (1.0 + math.exp(-sensitivity * delay_diff))
        
        # Stochastic Check
        if random.random() < prob:
            self.soft_drop_count += 1
            self.episode_soft_drops += 1
            return True # DROP
        return False # SURVIVE (for now)

    def step(self, action):
        finished_tasks = []

        # 1. Action Parsing (Same as before)
        iot_action_local = np.zeros(self.n_iot, dtype=int)
        iot_action_fog = np.zeros(self.n_iot, dtype=int)
        for i in range(self.n_iot):
            a = action[i]
            iot_action_fog[i] = int(a - 1)
            if a == 0:
                iot_action_local[i] = 1
            if self.bitArrive[self.time_count, i] > 0:
                self.total_tasks += 1
        # ---- Policy churn (Tier 3) ----
        if self.prev_actions is not None:
            churn = np.mean(action != self.prev_actions)
            self.log_policy_churn.append(churn)

        self.prev_actions = action.copy()


        # 2. Local Computation
        for i in range(self.n_iot):
            if self.bitArrive[self.time_count, i] > 0 and iot_action_local[i] == 1:
                self.Queue_iot_comp[i].put({'size': self.bitArrive[self.time_count, i], 'time': self.time_count})

            # Check Queue Drops (Soft Drop)
            if math.isnan(self.task_on_process_local[i]['remain']) and not self.Queue_iot_comp[i].empty():
                while not self.Queue_iot_comp[i].empty():
                    task = self.Queue_iot_comp[i].get()
                    if task['size'] == 0: continue
                    
                    current_delay = self.time_count - task['time'] + 1
                    # --- [MODIFIED] Soft Drop Check ---
                    if self._check_soft_drop(current_delay):
                        self.process_delay[task['time'], i] = self.max_delay # Log failure
                        self.process_delay_unfinish_ind[task['time'], i] = 1
                        r = self._calculate_reward(self.max_delay, 0, True)
                        finished_tasks.append({'iot': i, 'start_time': task['time'], 'reward': r, 'dropped': True, 'delay': current_delay, 'energy': 0.0})
                        self.drop_iot_count += 1
                        self.episode_return += r
                    else:
                        self.task_on_process_local[i] = {'size': task['size'], 'time': task['time'], 'remain': task['size']}
                        break

            # Process Active Task
            if not math.isnan(self.task_on_process_local[i]['remain']):
                capacity = self.comp_cap_iot[i] / self.comp_density[i]
                self.task_on_process_local[i]['remain'] -= capacity
                
                current_delay = self.time_count - self.task_on_process_local[i]['time'] + 1

                if self.task_on_process_local[i]['remain'] <= 0:
                    self.episode_success_count += 1
                    base_energy = self.coeff * (((self.comp_cap_iot[i]/self.duration)*1e9)**2) * \
              (self.comp_density[i]*1e9) * self.task_on_process_local[i]['size'] * 1e-2

                    local_load = 1.0 / self.n_iot   # single IoT load proxy

                    energy = self._nonlinear_comp_energy(local_load, base_energy)
                    self.episode_energy_sum += energy

                    self.process_delay[self.task_on_process_local[i]['time'], i] = current_delay
                    self.process_energy[self.task_on_process_local[i]['time'], i] = energy
                    r = self._calculate_reward(current_delay, energy, False)
                    finished_tasks.append({'iot': i, 'start_time': self.task_on_process_local[i]['time'], 'reward': r, 'dropped': False, 'delay': current_delay, 'energy': energy})
                    self.episode_return += r
                    self.task_on_process_local[i]['remain'] = np.nan
                
                # --- [MODIFIED] Soft Drop Check (During Processing) ---
                elif self._check_soft_drop(current_delay):
                    self.process_delay[self.task_on_process_local[i]['time'], i] = self.max_delay
                    self.process_delay_unfinish_ind[self.task_on_process_local[i]['time'], i] = 1
                    r = self._calculate_reward(self.max_delay, 0, True)
                    finished_tasks.append({'iot': i, 'start_time': self.task_on_process_local[i]['time'], 'reward': r, 'dropped': True, 'delay': current_delay, 'energy': 0.0})
                    self.episode_return += r
                    self.task_on_process_local[i]['remain'] = np.nan
                    self.drop_iot_count += 1

            # Update Wait Time Estimate
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

                    current_delay = self.time_count - task['time'] + 1
                    # --- [MODIFIED] Soft Drop Check ---
                    if self._check_soft_drop(current_delay):
                        self.process_delay[task['time'], i] = self.max_delay
                        self.process_delay_unfinish_ind[task['time'], i] = 1
                        r = self._calculate_reward(self.max_delay, 0, True)
                        finished_tasks.append({'iot': i, 'start_time': task['time'], 'reward': r, 'dropped': True, 'delay': current_delay, 'energy': 0.0})
                        self.drop_trans_count += 1
                        self.episode_return += r
                    else:
                        self.task_on_transmit_local[i] = {'size': task['size'], 'time': task['time'], 'fog': task['fog'], 'remain': task['size']}
                        break

            if not math.isnan(self.task_on_transmit_local[i]['remain']):
                f = int(self.task_on_transmit_local[i]['fog'])
                rate = self.tran_cap_sat if f == 0 else self.calc_tran(f, i)
                self.task_on_transmit_local[i]['remain'] -= rate

                if self.task_on_transmit_local[i]['remain'] <= 0:
                    # ---- base (linear) transmission energy ----
                    if f == 0:
                        rate = self.tran_cap_sat
                        base_e_trans = (
                            self.p_fog[0]
                            * self.task_on_transmit_local[i]['size']
                            / (rate / self.duration)
                        )
                    else:
                        rate = self.calc_tran(f, i)
                        base_e_trans = (
                            self.p_fog[f]
                            * self.task_on_transmit_local[i]['size']
                            / (rate / self.duration)
                        )

                    # ---- nonlinear transmission energy ----
                    max_rate = self.tran_cap_sat
                    e_trans = self._nonlinear_tx_energy(rate, max_rate, base_e_trans)
                    self.episode_energy_sum += e_trans
                    
                    self.Queue_fog_comp[i][f].put({
                        'size': self.task_on_transmit_local[i]['size'],
                        'time': self.task_on_transmit_local[i]['time'],
                        'energy_trans': e_trans
                    })
                    self.b_fog_comp[i][f] += self.task_on_transmit_local[i]['size']
                    self.task_on_transmit_local[i]['remain'] = np.nan
                
                else:
                    current_delay = self.time_count - self.task_on_transmit_local[i]['time'] + 1
                    # --- [MODIFIED] Soft Drop Check ---
                    if self._check_soft_drop(current_delay):
                        self.process_delay[self.task_on_transmit_local[i]['time'], i] = self.max_delay
                        self.process_delay_unfinish_ind[self.task_on_transmit_local[i]['time'], i] = 1
                        r = self._calculate_reward(self.max_delay, 0, True)
                        finished_tasks.append({'iot': i, 'start_time': self.task_on_transmit_local[i]['time'], 'reward': r, 'dropped': True, 'delay': current_delay, 'energy': 0.0})
                        self.task_on_transmit_local[i]['remain'] = np.nan
                        self.drop_trans_count += 1
                        self.episode_return += r

            # Update Wait Time Estimate
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

        # 4. Fog Computation (The Critical Section)
        for i in range(self.n_iot):
            for f in range(self.n_fog):
                # Queue Processing
                if math.isnan(self.task_on_process_fog[i][f]['remain']) and not self.Queue_fog_comp[i][f].empty():
                    while not self.Queue_fog_comp[i][f].empty():
                        task = self.Queue_fog_comp[i][f].get()
                        
                        current_delay = self.time_count - task['time'] + 1
                        # --- [MODIFIED] Soft Drop Check ---
                        if self._check_soft_drop(current_delay):
                            self.process_delay[task['time'], i] = self.max_delay
                            self.process_delay_unfinish_ind[task['time'], i] = 1
                            r = self._calculate_reward(self.max_delay, 0, True)
                            finished_tasks.append({'iot': i, 'start_time': task['time'], 'reward': r, 'dropped': True, 'delay': current_delay, 'energy': 0.0})
                            self.drop_fog_count += 1
                            self.episode_return += r
                            self.b_fog_comp[i][f] = max(0, self.b_fog_comp[i][f] - task['size'])
                        else:
                            self.task_on_process_fog[i][f] = {'size': task['size'], 'time': task['time'], 'remain': task['size'], 'energy': task['energy_trans']}
                            break

                # Active Processing
                # if not math.isnan(self.task_on_process_fog[i][f]['remain']):
                #     # --- [MODIFIED] Non-Linear Capacity (The Trap) ---
                #     # 1. Base share (1/N)
                #     share = self.fog_iot_m[f] if self.fog_iot_m[f] > 0 else 1
                    
                #     # 2. Get Congestion Penalty (The Hockey Stick)
                #     congestion_penalty = self._get_congestion_penalty(f)
                    
                #     # 3. Calculate Effective Capacity
                #     # Divides capability by share AND congestion penalty
                #     capacity = (self.comp_cap_fog[f] / self.comp_density[i]) / (share * congestion_penalty)
                
                # Active Processing
                if not math.isnan(self.task_on_process_fog[i][f]['remain']):

                    # 1. Fair share
                    share = self.fog_iot_m[f] if self.fog_iot_m[f] > 0 else 1

                    # 2. Local congestion (hockey stick)
                    local_penalty = self._get_congestion_penalty(f)

                    # 4. Effective capacity (NO mutation, NO state change)
                    capacity = (
                        self.comp_cap_fog[f] / self.comp_density[i]
                    ) / (share * local_penalty)

                    
                    self.task_on_process_fog[i][f]['remain'] -= capacity
                    self.b_fog_comp[i][f] = max(0, self.b_fog_comp[i][f] - capacity)

                    current_delay = self.time_count - self.task_on_process_fog[i][f]['time'] + 1

                    if self.task_on_process_fog[i][f]['remain'] <= 0:
                        self.episode_success_count += 1
                        delay = current_delay
                        base_comp_energy = self.coeff * (((self.comp_cap_fog[f]/self.duration)*1e9)**2) * \
                   (self.comp_density[i]*1e9) * self.task_on_process_fog[i][f]['size'] * 1e-2

                        fog_load = self.fog_iot_m[f] / max(1, self.n_iot)
                        comp_energy = self._nonlinear_comp_energy(fog_load, base_comp_energy)

                        total_energy = self.task_on_process_fog[i][f]['energy'] + comp_energy
                        self.episode_energy_sum += total_energy
                        self.process_delay[self.task_on_process_fog[i][f]['time'], i] = delay
                        self.process_energy[self.task_on_process_fog[i][f]['time'], i] = total_energy

                        r = self._calculate_reward(delay, total_energy, False)
                        finished_tasks.append({'iot': i, 'start_time': self.task_on_process_fog[i][f]['time'], 'reward': r, 'dropped': False, 'delay': delay, 'energy': total_energy})
                        self.episode_return += r
                        self.task_on_process_fog[i][f]['remain'] = np.nan
                    
                    # --- [MODIFIED] Soft Drop Check (During Fog Processing) ---
                    elif self._check_soft_drop(current_delay):
                        self.process_delay[self.task_on_process_fog[i][f]['time'], i] = self.max_delay
                        self.process_delay_unfinish_ind[self.task_on_process_fog[i][f]['time'], i] = 1
                        
                        self.fog_drop[i, f] = self.task_on_process_fog[i][f]['remain']
                        
                        r = self._calculate_reward(self.max_delay, 0, True)
                        finished_tasks.append({'iot': i, 'start_time': self.task_on_process_fog[i][f]['time'], 'reward': r, 'dropped': True, 'delay': current_delay, 'energy': 0.0})
                        self.episode_return += r
                        self.task_on_process_fog[i][f]['remain'] = np.nan
                        self.drop_fog_count += 1
                        self.b_fog_comp[i][f] = max(0, self.b_fog_comp[i][f] - self.fog_drop[i, f])

        # 5. Congestion Update
        self.fog_iot_m.fill(0)
        for f in range(self.n_fog):
            for i in range(self.n_iot):
                if (not self.Queue_fog_comp[i][f].empty()
                    or not math.isnan(self.task_on_process_fog[i][f]['remain'])
                    or self.b_fog_comp[i][f] > 0):
                    self.fog_iot_m[f] += 1

        self.fog_iot_m_observe = self.fog_iot_m.copy()
        # ---- Fog utilization variance ----
        fog_var = np.var(self.fog_iot_m)
        self._fog_var_accumulator.append(fog_var)

        # ---- Lyapunov proxy (Tier 3) ----
        if self._prev_fog_load is not None:
            delta = np.mean(np.abs(self.fog_iot_m - self._prev_fog_load))
            self.log_fog_delta.append(delta)

        self._prev_fog_load = self.fog_iot_m.copy()


        # 6. Update
        self.time_count += 1
        done = (self.time_count >= self.n_time)

        if done:
            finished_tasks.extend(self._finalize_episode())
            self.log_episode_return.append(self.episode_return)

            total_drops = self.episode_soft_drops + self.episode_hard_drops
            total_tasks = total_drops + self.episode_success_count
            drop_rate = total_drops / max(1, total_tasks)
            self.log_drop_rate.append(drop_rate)
            self.episode_return = 0.0
            self._prev_fog_load = None

            self.episode_soft_drops = 0
            self.episode_hard_drops = 0
            self.episode_success_count = 0
            self.episode_success_delay_sum = 0
            self.episode_near_miss_count = 0
            if self._fog_var_accumulator:
                mean_fog_var = float(np.mean(self._fog_var_accumulator))
            else:
                mean_fog_var = 0.0

            self.log_fog_var.append(mean_fog_var)
            self._fog_var_accumulator.clear()

            self.log_energy.append(self.episode_energy_sum)


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
                        self.episode_hard_drops += 1

            if not math.isnan(self.task_on_process_local[i]['remain']):
                t = self.task_on_process_local[i]['time']
                self.process_delay[t, i] = self.max_delay
                self.process_delay_unfinish_ind[t, i] = 1
                penalties.append({'iot': i, 'start_time': t, 'reward': r_drop, 'dropped': True, 'delay': self.max_delay, 'energy': 0.0})
                self.episode_hard_drops += 1

            if not math.isnan(self.task_on_transmit_local[i]['remain']):
                t = self.task_on_transmit_local[i]['time']
                self.process_delay[t, i] = self.max_delay
                self.process_delay_unfinish_ind[t, i] = 1
                penalties.append({'iot': i, 'start_time': t, 'reward': r_drop, 'dropped': True, 'delay': self.max_delay, 'energy': 0.0})
                self.episode_hard_drops += 1

            for f in range(self.n_fog):
                if not math.isnan(self.task_on_process_fog[i][f]['remain']):
                    t = self.task_on_process_fog[i][f]['time']
                    self.process_delay[t, i] = self.max_delay
                    self.process_delay_unfinish_ind[t, i] = 1
                    penalties.append({'iot': i, 'start_time': t, 'reward': r_drop, 'dropped': True, 'delay': self.max_delay, 'energy': 0.0})
                    self.episode_hard_drops += 1

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


