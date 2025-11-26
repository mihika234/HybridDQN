# brain.py
# Rewritten HybridDQN core - attention always enabled, robust diagnostics.
import os
import time
import math
import random
from collections import deque, Counter
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional quantum imports
try:
    import tensorcircuit as tc
    import tensorflow as tf
    TENSORCIRCUIT_AVAILABLE = True
except Exception:
    TENSORCIRCUIT_AVAILABLE = False

# -------------------------
# Small Attention wrapper
# -------------------------
class SimpleAttention(nn.Module):
    """
    Uses nn.MultiheadAttention with batch_first=True.
    Query = projected state vector (1 token), Key/Value = LSTM sequence tokens.
    Returns attended vector of same embedding dimension as query.
    """
    def __init__(self, embed_dim, num_heads=1):
        super(SimpleAttention, self).__init__()
        # keep a small number of heads by default
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        # a small layer to project LSTM outputs into embed_dim if needed
        self.key_proj = None
        self.value_proj = None
        self.embed_dim = embed_dim

    def ensure_projections(self, key_dim):
        if key_dim != self.embed_dim:
            # create/provide linear to convert key/value to embed_dim
            if self.key_proj is None:
                self.key_proj = nn.Linear(key_dim, self.embed_dim)
                self.value_proj = nn.Linear(key_dim, self.embed_dim)

    def forward(self, query, key_seq):
        """
        query: [B, Dq]  (we will unsqueeze to [B,1,Dq])
        key_seq: [B, T, Dk]
        returns: [B, embed_dim] attended vector
        """
        B, T, Dk = key_seq.shape
        q = query.unsqueeze(1)  # [B,1,Dq]
        # project keys/values if needed
        self.ensure_projections(Dk)
        if self.key_proj is not None:
            k = self.key_proj(key_seq)
            v = self.value_proj(key_seq)
        else:
            k = key_seq
            v = key_seq

        # use MHA: inputs are (B, L, E) with batch_first=True
        out, attn = self.mha(q, k, v, need_weights=False)
        # out shape [B,1,embed_dim]
        return out.squeeze(1)  # [B, embed_dim]

# -------------------------
# Quantum network (optional)
# -------------------------
class QuantumNetwork(nn.Module):
    """
    Thin wrapper for a PQC via tensorcircuit + classical head.
    Quantum layer is vmapped over the batch using tensorcircuit interfaces (if available).
    """
    def __init__(self,
                 n,              # number of qubits
                 layers,         # number of PQC layers
                 n_actions,
                 n_lstm,
                 n_lstm_state,
                 n_features,
                 n_l1,
                 batch_size,
                 seed,
                 non_sequential=False,
                 parallel_layers=1):
        super(QuantumNetwork, self).__init__()

        if not TENSORCIRCUIT_AVAILABLE:
            raise RuntimeError("TensorCircuit/TensorFlow not available. Cannot construct QuantumNetwork (set hybrid=False).")

        self.n = int(n)
        self.layers = int(layers)
        self.parallel_layers = int(parallel_layers)

        # backend
        self.K = tc.set_backend("tensorflow")

        # classical LSTM & dims
        self.n_lstm = int(n_lstm)
        self.batch_size = int(batch_size)
        self.state_dim = int(n_lstm + n_features)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.lstm = nn.LSTM(n_lstm_state, n_lstm, batch_first=True)

        # quantum parameters and layer wrapper
        if not non_sequential:
            # shape (layers, n_qubits, 3)
            self.weights = torch.nn.Parameter(torch.empty(self.layers, self.n, 3))
            # wrap vqc with tc interfaces (vmapped over batch)
            # note: the actual vqc function uses tf ops and constructs a fresh circuit per call
            self.quantum_layer = tc.interfaces.torch_interface(
                self.K.vmap(self.vqc, vectorized_argnums=0), jit=True
            )
        else:
            # keep signature; fallback to simple sequential mode if used
            self.weights = torch.nn.Parameter(torch.empty(self.parallel_layers, self.layers, self.n, 3))
            self.quantum_layer = tc.interfaces.torch_interface(
                self.K.vmap(self.non_sequential_vqc, vectorized_argnums=0), jit=True
            )

        # initialize weights to [0, 2pi)
        nn.init.uniform_(self.weights, 0, 2 * math.pi)

        # Attention: project LSTM outputs to a state_dim then attend
        self.attn = SimpleAttention(embed_dim=self.state_dim, num_heads=1)
        self.lstm_to_stateproj = nn.Linear(self.n_lstm, self.state_dim)

        # classical head consumes [state_full, q_features]
        self.nn_output = nn.Sequential(
            nn.Linear(self.state_dim + self.n, n_l1),
            nn.ReLU()
        )
        self.output = nn.Linear(n_l1, n_actions)

        # initialize linear layers consistently
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.3)
                nn.init.constant_(m.bias, 0.1)

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    @tf.function
    def vqc(self, state, weights):
        """
        state: 1D tf tensor (D)
        weights: tf tensor (layers, n, 3)
        returns: tf tensor (n) real expectation of Z per qubit
        This function uses a freshly constructed circuit each call (safe for autodiff).
        """
        D = self.K.shape_tensor(state)[0]
        c = tc.Circuit(self.n)

        # optional hadamards
        for q in range(self.n):
            c.h(q)

        # data reupload + param layers
        for layer in range(self.layers):
            for q in range(self.n):
                idx = (layer * self.n + q) % D
                angle = state[idx]
                c.rx(q, theta=angle)
                c.ry(q, theta=angle)
                c.rz(q, theta=weights[layer, q, 0])
                c.ry(q, theta=weights[layer, q, 1])
                c.rz(q, theta=weights[layer, q, 2])
            # ring entanglement
            for q in range(self.n - 1):
                c.cnot(q, q + 1)
            if self.n > 1:
                c.cnot(self.n - 1, 0)

        output = self.K.stack([self.K.real(c.expectation((tc.gates.z(), [q]))) for q in range(self.n)])
        return output

    @tf.function
    def non_sequential_vqc(self, inputs, weights):
        raise NotImplementedError("Non-sequential VQC mode is not implemented in this build.")

    def forward(self, s, lstm_s):
        """
        s: [B, n_features]  (numpy or torch)
        lstm_s: [B, T, n_lstm_state]
        returns: [B, n_actions]
        """
        # convert and device-placement
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float().to(self.device)
        if isinstance(lstm_s, np.ndarray):
            lstm_s = torch.from_numpy(lstm_s).float().to(self.device)

        # run LSTM (get sequence)
        h_0 = torch.zeros(1, lstm_s.shape[0], self.n_lstm, device=lstm_s.device)
        c_0 = torch.zeros(1, lstm_s.shape[0], self.n_lstm, device=lstm_s.device)
        lstm_outputs, _ = self.lstm(lstm_s, (h_0, c_0))  # [B, T, n_lstm]
        lstm_last = lstm_outputs[:, -1, :].reshape(-1, self.n_lstm)

        # state_full = concat(last_lstm, s) -> used as query
        state_full = torch.cat([lstm_last, s.to(lstm_last.device)], dim=1)  # [B, D]
        # project lstm outputs to match state_dim for attention
        lstm_proj = self.lstm_to_stateproj(lstm_outputs)  # [B, T, state_dim]
        # attention (query = state_full, key/value = lstm_proj)
        attn_out = self.attn(state_full, lstm_proj)  # [B, state_dim]
        # merge: use attn_out as refined state representation
        merged_state = torch.cat([attn_out, s.to(attn_out.device)], dim=1)  # [B, state_dim + n_features]
        # normalize angles into [-pi, pi]
        state_norm = torch.tanh(merged_state) * math.pi
        # quantum layer expects TF-backed arrays; tc's torch_interface handles conversion
        try:
            q_features = self.quantum_layer(state_norm, self.weights)  # [B, n_qubits]
        except Exception as e:
            # if the quantum layer fails for a batch, raise with informative message
            raise RuntimeError("Quantum forward failed: {}".format(repr(e)))
        merged = torch.cat([merged_state, q_features.to(merged_state.device)], dim=1)
        x = self.nn_output(merged)
        return self.output(x)

    def save_weights(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "quantum_net_state.pt"))
        try:
            w = self.weights.detach().cpu().numpy()
            np.save(os.path.join(path, "pqc_weights.npy"), w)
        except Exception:
            pass

# -------------------------
# Classical network
# -------------------------
class ClassicalNetwork(nn.Module):
    def __init__(self, n_actions, n_features, n_lstm_state, n_l1, n_lstm, batch_size, seed):
        super(ClassicalNetwork, self).__init__()

        self.n_lstm = int(n_lstm)
        self.batch_size = int(batch_size)
        self.state_dim = int(n_lstm + n_features)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.lstm = nn.LSTM(n_lstm_state, n_lstm, batch_first=True)
        self.relu = nn.ReLU()

        # attention: use last LSTM output as query, sequence as keys
        self.lstm_to_stateproj = nn.Linear(self.n_lstm, self.state_dim)
        self.attn = SimpleAttention(embed_dim=self.state_dim, num_heads=1)

        self.fc1 = nn.Linear(self.state_dim + n_features, n_l1)
        self.fc12 = nn.Linear(n_l1, n_l1)
        self.output = nn.Linear(n_l1, n_actions)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=0.3)
                nn.init.constant_(m.bias, 0.1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, s, lstm_s):
        """
        s: [B, n_features]
        lstm_s: [B, T, n_lstm_state]
        """
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float().to(self.device)
        if isinstance(lstm_s, np.ndarray):
            lstm_s = torch.from_numpy(lstm_s).float().to(self.device)

        h_0 = torch.zeros(1, lstm_s.shape[0], self.n_lstm, device=lstm_s.device)
        c_0 = torch.zeros(1, lstm_s.shape[0], self.n_lstm, device=lstm_s.device)
        lstm_outputs, _ = self.lstm(lstm_s, (h_0, c_0))  # [B, T, n_lstm]
        lstm_last = lstm_outputs[:, -1, :].reshape(-1, self.n_lstm)

        # project lstm seq to match attention embed dim
        lstm_proj = self.lstm_to_stateproj(lstm_outputs)  # [B, T, state_dim]
        # state_full query
        state_full = torch.cat([lstm_last, s.to(lstm_last.device)], dim=1)  # [B, D]
        attn_out = self.attn(state_full, lstm_proj)  # [B, state_dim]
        merged = torch.cat([attn_out, s.to(attn_out.device)], dim=1)  # [B, state_dim + n_features]

        x = self.relu(self.fc1(merged))
        x = self.relu(self.fc12(x))
        return self.output(x)

# -------------------------
# Hybrid DQN manager (safe, diagnostic)
# -------------------------
class HybridDQN:
    def __init__(self,
                 n_actions,
                 n_features,
                 n_lstm_features,
                 n_time,
                 learning_rate=0.001,
                 reward_decay=0.9,
                 e_greedy=0.99,
                 replace_target_iter=200,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=0.00025,
                 n_lstm_step=10,
                 N_L1=20,
                 N_lstm=20,
                 optimizer='rms_prop',
                 seed=0,
                 hybrid=False,
                 qubits=3,
                 layers=3,
                 non_sequential=False,
                 parallel_layers=1,
                 training_dir='training/',
                 weights=None):
        """
        HybridDQN manager. Always uses attention. Hybrid=True will attempt to build a quantum net.
        """
        self.hybrid = bool(hybrid)
        self.training_dir = training_dir if training_dir is not None else "training/"
        os.makedirs(self.training_dir, exist_ok=True)

        # diagnostics path
        self.diag_path = os.path.join(self.training_dir, "diagnostics_global.txt")
        with open(self.diag_path, "a") as f:
            f.write("\n\n==== New training session started: {} ====\n".format(time.asctime()))

        # core parameters
        self.n_actions = int(n_actions)
        self.n_features = int(n_features)
        self.n_time = int(n_time)
        self.gamma = float(reward_decay)
        self.epsilon_max = float(e_greedy)
        self.replace_target_iter = int(replace_target_iter)
        self.memory_size = int(memory_size)
        self.batch_size = int(batch_size)
        self.epsilon_increment = float(e_greedy_increment) if e_greedy_increment is not None else None
        self.epsilon = 0.0 if self.epsilon_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.N_L1 = int(N_L1)
        self.N_lstm = int(N_lstm)
        self.n_lstm_step = int(n_lstm_step)
        self.n_lstm_state = int(n_lstm_features)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if optimizer not in ['adam', 'rms_prop', 'gd']:
            raise SystemExit("Invalid optimizer: {}. Choose one of ['adam','rms_prop','gd']".format(optimizer))

        # build nets
        if self.hybrid:
            if not TENSORCIRCUIT_AVAILABLE:
                raise RuntimeError("Requested hybrid network but tensorcircuit/tf not available in environment.")
            self.target_net = QuantumNetwork(qubits, layers, self.n_actions, N_lstm, self.n_lstm_state,
                                             self.n_features, self.N_L1, self.batch_size, seed, non_sequential, parallel_layers)
            self.eval_net = QuantumNetwork(qubits, layers, self.n_actions, N_lstm, self.n_lstm_state,
                                           self.n_features, self.N_L1, self.batch_size, seed, non_sequential, parallel_layers)
        else:
            self.target_net = ClassicalNetwork(self.n_actions, self.n_features, self.n_lstm_state, self.N_L1, self.N_lstm, self.batch_size, seed)
            self.eval_net = ClassicalNetwork(self.n_actions, self.n_features, self.n_lstm_state, self.N_L1, self.N_lstm, self.batch_size, seed)

        # optim
        if optimizer == 'adam':
            self.eval_optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        elif optimizer == 'gd':
            self.eval_optimizer = torch.optim.SGD(self.eval_net.parameters(), lr=learning_rate)
        elif optimizer == 'rms_prop':
            self.eval_optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=learning_rate)

        self.loss_fn = nn.MSELoss()

        # replay memory as list of dicts
        self.memory = []
        self.memory_counter = 0
        self.max_memory_size = self.memory_size

        # stores
        self.reward_store = []
        self.action_store = []
        self.delay_store = []
        self.energy_store = []

        self.lstm_history = deque(maxlen=self.n_lstm_step)
        for _ in range(self.n_lstm_step):
            self.lstm_history.append(torch.zeros(self.n_lstm_state).to(self.device))

        self.store_q_value = []

        # initial diag entry
        self._log_diag("Initialized HybridDQN: hybrid=%s device=%s batch_size=%d n_lstm_step=%d" %
                       (self.hybrid, str(self.device), self.batch_size, self.n_lstm_step))

    # -------------------------
    # Diagnostics utilities
    # -------------------------
    def _log_diag(self, msg):
        ts = time.asctime()
        line = "[{}] {}\n".format(ts, msg)
        try:
            with open(self.diag_path, "a") as f:
                f.write(line)
        except Exception:
            # fallback print
            print("DIAG_WRITE_FAIL:", line)

    def _diag_batch_stats(self, q_eval=None, q_target_next=None, loss=None, grads=None, pqc_weights=None, actions=None, rewards=None):
        try:
            if loss is not None:
                try:
                    self._log_diag("loss: {:.6e}".format(float(loss.item())))
                except Exception:
                    self._log_diag(f"loss: {loss}")
            if q_eval is not None:
                try:
                    qeval_np = q_eval.detach().cpu().numpy()
                    self._log_diag("q_eval mean={:.6e} std={:.6e} min={:.6e} max={:.6e}".format(qeval_np.mean(), qeval_np.std(), qeval_np.min(), qeval_np.max()))
                except Exception as e:
                    self._log_diag("[diag-qeval-error] " + repr(e))
            if q_target_next is not None:
                try:
                    qtn = q_target_next.detach().cpu().numpy()
                    self._log_diag("q_target_next mean={:.6e} std={:.6e} min={:.6e} max={:.6e}".format(qtn.mean(), qtn.std(), qtn.min(), qtn.max()))
                except Exception as e:
                    self._log_diag("[diag-qtarget-error] " + repr(e))
            if grads is not None:
                grad_str = " ".join(["{:.3e}".format(g) for g in grads])
                self._log_diag("param_grad_norms: {}".format(grad_str))
                if len(grads) > 0:
                    avg = float(np.mean(grads))
                    self._log_diag("global_avg_grad_norm: {:.6e}".format(avg))
                    if self.hybrid and avg < 1e-7:
                        self._log_diag("WARNING: very small global avg grad norm -> possible barren plateau or vanishing gradients")
            if pqc_weights is not None:
                try:
                    w = pqc_weights.detach().cpu().numpy()
                    self._log_diag("pqc_weights mean={:.6e} std={:.6e}".format(w.mean(), w.std()))
                except Exception as e:
                    self._log_diag("[diag-pqc-err] " + repr(e))
            if actions is not None:
                hist = Counter([int(a) for a in actions])
                self._log_diag("action_hist: {}".format(dict(hist)))
            if rewards is not None:
                r = np.array(rewards)
                self._log_diag("reward_stats mean={:.6e} std={:.6e} min={:.6e} max={:.6e}".format(r.mean(), r.std(), r.min(), r.max()))
            # memory health
            self._log_diag("memory_size: {} memory_counter: {}".format(len(self.memory), self.memory_counter))
        except Exception as e:
            with open(self.diag_path, "a") as f:
                f.write("[diag-error] {}\n".format(repr(e)))

    # -------------------------
    # Memory API (safe)
    # -------------------------
    def store_transition(self, s, lstm_s, a, r, s_, lstm_s_):
        """
        Store transition as a dict. Accepts numpy arrays or torch tensors.
        """
        try:
            # convert to numpy for consistent storage
            if isinstance(s, torch.Tensor):
                s = s.detach().cpu().numpy()
            if isinstance(s_, torch.Tensor):
                s_ = s_.detach().cpu().numpy()
            if isinstance(lstm_s, torch.Tensor):
                lstm_s = lstm_s.detach().cpu().numpy()
            if isinstance(lstm_s_, torch.Tensor):
                lstm_s_ = lstm_s_.detach().cpu().numpy()

            transition = {
                's': np.array(s, copy=True),
                'a': int(np.int64(a)),
                'r': float(r),
                's_': np.array(s_, copy=True),
                'lstm': np.array(lstm_s, copy=True),
                'lstm_': np.array(lstm_s_, copy=True),
                'time': time.time()
            }
            if len(self.memory) >= self.max_memory_size:
                # replace oldest to keep memory bounded
                self.memory[self.memory_counter % self.max_memory_size] = transition
            else:
                self.memory.append(transition)
            self.memory_counter += 1
        except Exception as e:
            self._log_diag("[store_transition_error] {}".format(repr(e)))

    # -------------------------
    # LSTM history update (for online action selection)
    # -------------------------
    def update_lstm(self, lstm_s):
        if isinstance(lstm_s, np.ndarray):
            t = torch.from_numpy(lstm_s).float().to(self.device)
        elif isinstance(lstm_s, torch.Tensor):
            t = lstm_s.float().to(self.device)
        else:
            t = torch.tensor(lstm_s, dtype=torch.float32, device=self.device)
        # keep last element shape consistent with n_lstm_state
        if t.numel() != self.n_lstm_state:
            v = t.view(-1).detach().cpu().numpy()
            if v.size < self.n_lstm_state:
                v = np.pad(v, (0, self.n_lstm_state - v.size), 'constant')
            else:
                v = v[:self.n_lstm_state]
            t = torch.from_numpy(v).float().to(self.device)
        self.lstm_history.append(t)

    # -------------------------
    # Action selection
    # -------------------------
    def choose_action(self, observation, inference=False):
        # observation: 1D array-like of n_features
        obs = np.asarray(observation, dtype=np.float32)
        obs = obs[np.newaxis, :]  # [1, n_features]

        # build lstm batch: [1, T, n_lstm_state]
        lstm_observation = torch.stack([t for t in self.lstm_history], dim=0).to(self.device).float()
        lstm_batch = lstm_observation.reshape(1, self.n_lstm_step, self.n_lstm_state)

        # epsilon-greedy: exploration if random number < epsilon (unless inference=True)
        if (not inference) and (np.random.uniform() < self.epsilon):
            return int(np.random.randint(0, self.n_actions))

        # else evaluate policy
        try:
            with torch.no_grad():
                q_values = self.eval_net(obs, lstm_batch)  # expects torch tensors / numpy
                if isinstance(q_values, torch.Tensor):
                    action = int(torch.argmax(q_values[0]).cpu().item())
                else:
                    qvals_np = np.array(q_values)
                    action = int(np.argmax(qvals_np[0]))
                # store q-value snapshot if possible
                try:
                    q_np = q_values.detach().cpu().numpy() if isinstance(q_values, torch.Tensor) else np.array(q_values)
                    self.store_q_value.append({'observation': obs, 'q_value': q_np})
                    if len(self.store_q_value) > 1000:
                        self.store_q_value.pop(0)
                except Exception:
                    pass
                return action
        except Exception as e:
            self._log_diag("[choose_action_error] {}".format(repr(e)))
            return int(np.random.randint(0, self.n_actions))

    def store_q_snapshot(self, obs, q_values):
        try:
            self.store_q_value.append({'obs': obs, 'q': q_values, 'step': self.learn_step_counter})
            if len(self.store_q_value) > 1000:
                self.store_q_value.pop(0)
        except Exception:
            pass

    # -------------------------
    # Target update helper
    # -------------------------
    def update_target_net(self):
        try:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            self._log_diag("target_net updated at step {}".format(self.learn_step_counter))
        except Exception as e:
            self._log_diag("[target_update_error] {}".format(repr(e)))

    # -------------------------
    # Learning (core)
    # -------------------------
    def learn(self):
        """
        Core training step. Samples sequential chunks of length n_lstm_step from memory.
        Writes diagnostics to file every call.
        """
        # ensure enough memory
        if len(self.memory) < max(self.batch_size, self.n_lstm_step + 1):
            self._log_diag("learn() early return: not enough memory (len_memory=%d, need=%d)" %
                           (len(self.memory), max(self.batch_size, self.n_lstm_step + 1)))
            return

        # update target net periodically
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.update_target_net()

        # sample starting indices for sequences of length n_lstm_step
        max_start = len(self.memory) - self.n_lstm_step
        if max_start <= 0:
            self._log_diag("learn() early return: max_start <= 0 (max_start=%d)" % max_start)
            return
        start_indices = np.random.choice(max_start, size=self.batch_size, replace=(max_start < self.batch_size))

        # prepare batches
        batch_s = []
        batch_s_ = []
        batch_actions = []
        batch_rewards = []
        batch_lstm_current = np.zeros((self.batch_size, self.n_lstm_step, self.n_lstm_state), dtype=np.float32)
        batch_lstm_next = np.zeros((self.batch_size, self.n_lstm_step, self.n_lstm_state), dtype=np.float32)

        for bi, start in enumerate(start_indices):
            seq = self.memory[start: start + self.n_lstm_step]
            if len(seq) < self.n_lstm_step:
                seq = seq + [seq[-1]] * (self.n_lstm_step - len(seq))
            last = seq[-1]
            batch_s.append(last['s'])
            batch_s_.append(last['s_'])
            batch_actions.append(last['a'])
            batch_rewards.append(last['r'])
            for t_idx in range(self.n_lstm_step):
                row = seq[t_idx]
                lstm_cur = np.asarray(row.get('lstm'), dtype=np.float32).reshape(-1)
                lstm_nxt = np.asarray(row.get('lstm_'), dtype=np.float32).reshape(-1)
                if lstm_cur.size < self.n_lstm_state:
                    lstm_cur = np.pad(lstm_cur, (0, self.n_lstm_state - lstm_cur.size), 'constant')
                else:
                    lstm_cur = lstm_cur[:self.n_lstm_state]
                if lstm_nxt.size < self.n_lstm_state:
                    lstm_nxt = np.pad(lstm_nxt, (0, self.n_lstm_state - lstm_nxt.size), 'constant')
                else:
                    lstm_nxt = lstm_nxt[:self.n_lstm_state]
                batch_lstm_current[bi, t_idx, :] = lstm_cur
                batch_lstm_next[bi, t_idx, :] = lstm_nxt

        try:
            s_batch = torch.from_numpy(np.stack(batch_s)).float().to(self.device)
            s_next_batch = torch.from_numpy(np.stack(batch_s_)).float().to(self.device)
            actions_batch = torch.from_numpy(np.array(batch_actions, dtype=np.int64)).to(self.device)
            rewards_batch = torch.from_numpy(np.array(batch_rewards, dtype=np.float32)).to(self.device)
            lstm_cur_batch = torch.from_numpy(batch_lstm_current).float().to(self.device)
            lstm_next_batch = torch.from_numpy(batch_lstm_next).float().to(self.device)
        except Exception as e:
            self._log_diag("[batch_conversion_error] {}".format(repr(e)))
            return

        # forward passes
        try:
            q_eval = self.eval_net(s_batch, lstm_cur_batch)        # [B, n_actions]
        except Exception as e:
            self._log_diag("[forward_eval_error] {}".format(repr(e)))
            return

        try:
            q_target_next = self.target_net(s_next_batch, lstm_next_batch)  # [B, n_actions]
        except Exception as e:
            self._log_diag("[forward_target_error] {}".format(repr(e)))
            q_target_next = None

        # Build q_target using Double-DQN style
        with torch.no_grad():
            if q_target_next is not None:
                try:
                    q_next_selected = q_target_next.max(dim=1)[0]  # [B]
                    q_target_values = rewards_batch + self.gamma * q_next_selected
                except Exception as e:
                    self._log_diag("[qtarget_compute_error] {}".format(repr(e)))
                    q_target_values = rewards_batch.clone()
            else:
                q_target_values = rewards_batch.clone()

        # prepare q_target tensor (copy from q_eval then replace selected actions)
        q_eval_clone = q_eval.clone()
        batch_indices = torch.arange(self.batch_size, dtype=torch.int64).to(self.device)
        try:
            q_eval_clone[batch_indices, actions_batch] = q_target_values
            assign_ok = True
        except Exception as e:
            assign_ok = False
            self._log_diag("[q_target_assign_error] {}".format(repr(e)))

        if assign_ok:
            loss = self.loss_fn(q_eval_clone, q_eval)
        else:
            try:
                q_eval_selected = q_eval[batch_indices, actions_batch]
                loss = torch.mean((q_eval_selected - q_target_values) ** 2)
            except Exception as e:
                self._log_diag("[fallback_loss_error] {}".format(repr(e)))
                loss = torch.tensor(0.0, requires_grad=True).to(self.device)

        # diagnostics before backward
        try:
            pqc_weights = None
            if self.hybrid and hasattr(self.eval_net, 'weights'):
                try:
                    pqc_weights = self.eval_net.weights.detach().cpu()
                except Exception:
                    pqc_weights = None

            self._diag_batch_stats(q_eval=q_eval, q_target_next=q_target_next, loss=loss,
                                   grads=None, pqc_weights=pqc_weights,
                                   actions=actions_batch.detach().cpu().numpy() if actions_batch is not None else None,
                                   rewards=rewards_batch.detach().cpu().numpy() if rewards_batch is not None else None)
        except Exception as e:
            self._log_diag("[diag_before_backward_error] {}".format(repr(e)))

        self.eval_optimizer.zero_grad()
        try:
            loss.backward()
            # gradient norms
            grad_norms = []
            for p in self.eval_net.parameters():
                if p.grad is not None:
                    try:
                        grad_norms.append(float(p.grad.detach().cpu().norm().item()))
                    except Exception:
                        grad_norms.append(0.0)
                else:
                    grad_norms.append(0.0)
            torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=5.0)
            try:
                self._diag_batch_stats(q_eval=None, q_target_next=None, loss=None,
                                       grads=grad_norms, pqc_weights=pqc_weights,
                                       actions=None, rewards=None)
            except Exception:
                pass
            self.eval_optimizer.step()
        except Exception as e:
            self._log_diag("[backprop_error] {}".format(repr(e)))
            return

        # optionally save PQC weights periodically
        if self.hybrid:
            if (self.learn_step_counter % 200) == 0:
                try:
                    self.eval_net.save_weights(os.path.join(self.training_dir, "weights"))
                    self._log_diag("Saved hybrid network weights at step {}".format(self.learn_step_counter))
                except Exception as e:
                    self._log_diag("[save_weights_error] {}".format(repr(e)))

        # increment counters
        if self.epsilon_increment is not None:
            self.epsilon = min(self.epsilon + self.epsilon_increment, self.epsilon_max)
        self.learn_step_counter += 1

    # -------------------------
    # Convenience storage functions used by train.py
    # -------------------------
    def do_store_reward(self, episode, time_idx, reward):
        while episode >= len(self.reward_store):
            self.reward_store.append(torch.zeros(self.n_time).to(self.device))
        self.reward_store[episode][time_idx] = reward

    def do_store_action(self, episode, time_idx, action):
        while episode >= len(self.action_store):
            self.action_store.append(- torch.ones(self.n_time).to(self.device))
        self.action_store[episode][time_idx] = action

    def do_store_delay(self, episode, time_idx, delay):
        while episode >= len(self.delay_store):
            self.delay_store.append(np.zeros([self.n_time]))
        self.delay_store[episode][time_idx] = delay

    def do_store_energy(self, episode, time_idx, energy):
        while episode >= len(self.energy_store):
            self.energy_store.append(np.zeros([self.n_time]))
        self.energy_store[episode][time_idx] = energy

    # -------------------------
    # Save/Load helpers
    # -------------------------
    def save(self, path=None):
        path = path or self.training_dir
        os.makedirs(path, exist_ok=True)
        try:
            torch.save(self.eval_net.state_dict(), os.path.join(path, "eval_net_state.pt"))
            torch.save(self.target_net.state_dict(), os.path.join(path, "target_net_state.pt"))
            if self.hybrid and hasattr(self.eval_net, 'weights'):
                try:
                    np.save(os.path.join(path, "pqc_weights.npy"), self.eval_net.weights.detach().cpu().numpy())
                except Exception:
                    pass
            self._log_diag("[save] model saved to {}".format(path))
        except Exception as e:
            self._log_diag("[save_error] {}".format(repr(e)))

    def load(self, path):
        try:
            self.eval_net.load_state_dict(torch.load(os.path.join(path, "eval_net_state.pt"), map_location=self.device))
            self.target_net.load_state_dict(torch.load(os.path.join(path, "target_net_state.pt"), map_location=self.device))
            if self.hybrid and hasattr(self.eval_net, 'weights'):
                wpath = os.path.join(path, "pqc_weights.npy")
                if os.path.exists(wpath):
                    w = np.load(wpath)
                    try:
                        with torch.no_grad():
                            self.eval_net.weights.copy_(torch.from_numpy(w).to(self.eval_net.weights.device))
                            self.target_net.weights.copy_(torch.from_numpy(w).to(self.target_net.weights.device))
                    except Exception:
                        pass
            self._log_diag("[load] model loaded from {}".format(path))
        except Exception as e:
            self._log_diag("[load_error] {}".format(repr(e)))

# End of file
