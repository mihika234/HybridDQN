
import os
import time
import math
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Optional quantum imports
# =========================
try:
    import tensorcircuit as tc
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    TENSORCIRCUIT_AVAILABLE = True
except Exception:
    tc = None
    tf = None
    TENSORCIRCUIT_AVAILABLE = False


# ============================================================
# Attention (NO residual)
# ============================================================
class SimpleAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.key_proj = None
        self.value_proj = None
        self.ln = nn.LayerNorm(embed_dim)

    def _ensure_proj(self, key_dim, device):
        if key_dim != self.embed_dim and self.key_proj is None:
            self.key_proj = nn.Linear(key_dim, self.embed_dim).to(device)
            self.value_proj = nn.Linear(key_dim, self.embed_dim).to(device)

    def forward(self, query, key_seq):
        self._ensure_proj(key_seq.shape[-1], query.device)
        q = query.unsqueeze(1)
        k = self.key_proj(key_seq) if self.key_proj else key_seq
        v = self.value_proj(key_seq) if self.value_proj else key_seq
        out, _ = self.mha(q, k, v, need_weights=False)
        return self.ln(out.squeeze(1))


# ============================================================
# Quantum Network
# ============================================================
if TENSORCIRCUIT_AVAILABLE:
    class QuantumNetwork(nn.Module):
        def __init__(self, n_qubits, layers, n_actions, n_lstm, n_lstm_state, n_features, n_l1, batch_size, seed, training_dir):
            super().__init__()
            torch.manual_seed(seed)
            self.n = int(n_qubits)
            self.layers = int(layers)
            self.n_lstm = int(n_lstm)
            self.state_dim = self.n_lstm + n_features
            self.training_dir = training_dir
            self.K = tc.set_backend("tensorflow")
            self.lstm = nn.LSTM(n_lstm_state, self.n_lstm, batch_first=True)
            self.weights = nn.Parameter(torch.empty(self.layers, self.n, 3))
            nn.init.uniform_(self.weights, 0.0, 2.0 * math.pi)
            self.fc1 = nn.Linear(self.state_dim + self.n, n_l1)
            self.fc2 = nn.Linear(n_l1, n_actions)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(self.device)
            self.quantum_layer = tc.interfaces.torch_interface(self.K.vmap(self.vqc, vectorized_argnums=0), jit=True)
            self._save_circuit()
            self.loss_store = []



        @tf.function
        def vqc(self, state, weights):
            D = self.K.shape_tensor(state)[0]
            c = tc.Circuit(self.n)
            for layer in range(self.layers):
                for q in range(self.n):
                    idx = (layer * self.n + q) % D
                    a = state[idx]
                    c.rx(q, theta=a)
                    c.ry(q, theta=a)
                    c.rz(q, theta=weights[layer, q, 0])
                    c.ry(q, theta=weights[layer, q, 1])
                    c.rz(q, theta=weights[layer, q, 2])
                if self.n > 1:
                    for q in range(self.n - 1):
                        c.cnot(q, q + 1)
                    c.cnot(self.n - 1, 0)
            return self.K.stack([self.K.real(c.expectation((tc.gates.z(), [q]))) for q in range(self.n)])

        def _save_circuit(self):
            os.makedirs(self.training_dir, exist_ok=True)
            print("\n[Quantum Circuit Initialized]")

        def forward(self, s, lstm_s):
            if isinstance(s, np.ndarray): s = torch.tensor(s, dtype=torch.float32, device=self.device)
            if isinstance(lstm_s, np.ndarray): lstm_s = torch.tensor(lstm_s, dtype=torch.float32, device=self.device)
            if s.dim() == 1: s = s.unsqueeze(0)
            if lstm_s.dim() == 2: lstm_s = lstm_s.unsqueeze(0)
            h0 = torch.zeros(1, lstm_s.size(0), self.n_lstm, device=self.device)
            c0 = torch.zeros_like(h0)
            lstm_out, _ = self.lstm(lstm_s, (h0, c0))
            lstm_last = lstm_out[:, -1, :]
            state = torch.cat([lstm_last, s], dim=1)
            angles = torch.tanh(state) * math.pi
            q_feat = self.quantum_layer(angles, self.weights).to(self.device).float()
            x = torch.cat([state, q_feat], dim=1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)
else:
    QuantumNetwork = None


# ============================================================
# Classical Network
# ============================================================
class ClassicalNetwork(nn.Module):
    def __init__(self, n_actions, n_features, n_lstm_state, n_l1, n_lstm, batch_size, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.n_lstm = n_lstm
        self.state_dim = n_lstm + n_features
        self.lstm = nn.LSTM(n_lstm_state, n_lstm, batch_first=True)
        self.lstm_proj = nn.Linear(self.n_lstm, self.state_dim)
        self.attn = SimpleAttention(self.state_dim)
        self.fc1 = nn.Linear(self.state_dim + n_features, n_l1)
        self.fc2 = nn.Linear(n_l1, n_l1)
        self.output = nn.Linear(n_l1, n_actions)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, s, lstm_s):
        if isinstance(s, np.ndarray): s = torch.tensor(s, dtype=torch.float32, device=self.device)
        if isinstance(lstm_s, np.ndarray): lstm_s = torch.tensor(lstm_s, dtype=torch.float32, device=self.device)
        if s.dim() == 1: s = s.unsqueeze(0)
        if lstm_s.dim() == 2: lstm_s = lstm_s.unsqueeze(0)
        h0 = torch.zeros(1, lstm_s.size(0), self.n_lstm, device=self.device)
        c0 = torch.zeros_like(h0)
        lstm_out, _ = self.lstm(lstm_s, (h0, c0))
        lstm_last = lstm_out[:, -1, :]
        lstm_proj = self.lstm_proj(lstm_out)
        state = torch.cat([lstm_last, s], dim=1)
        attn_out = self.attn(state, lstm_proj)
        x = torch.cat([attn_out, s], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)


# ============================================================
# Hybrid DQN Manager
# ============================================================
class HybridDQN:
    def __init__(
        self, n_actions, n_features, n_lstm_features, n_time,
        learning_rate=0.001, reward_decay=0.9, e_greedy=1.0,
        replace_target_iter=200, memory_size=10000, batch_size=32,
        e_greedy_increment=0.0005, n_lstm_step=10,
        N_L1=64, N_lstm=32,
        optimizer="rms_prop", seed=0,
        hybrid=False, qubits=3, layers=3,
        training_dir="training/"
    ):
        self.hybrid = hybrid
        self.training_dir = training_dir
        os.makedirs(training_dir, exist_ok=True)

        self.n_actions = n_actions
        self.n_features = n_features
        self.n_time = n_time
        self.gamma = reward_decay
        self.batch_size = batch_size

        self.epsilon = e_greedy
        self.epsilon_min = 0.01
        target_decay_episodes = 800 
        self.epsilon_decay = (1.0 - 0.01) / target_decay_episodes

        self.replace_target_iter = replace_target_iter
        self.learn_step_counter = 0
        self.buffer_size_store = []
        self.loss_store = []
        self.grad_norm_store = []
        self.q_mean_store = []
        self.q_std_store = []

        self.n_lstm_step = n_lstm_step
        self.n_lstm_state = n_lstm_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if hybrid:
            if QuantumNetwork is None: raise ImportError("TensorCircuit not available.")
            self.eval_net = QuantumNetwork(qubits, layers, n_actions, N_lstm, n_lstm_features, n_features, N_L1, batch_size, seed, training_dir)
            self.target_net = QuantumNetwork(qubits, layers, n_actions, N_lstm, n_lstm_features, n_features, N_L1, batch_size, seed, training_dir)
        else:
            self.eval_net = ClassicalNetwork(n_actions, n_features, n_lstm_features, N_L1, N_lstm, batch_size, seed)
            self.target_net = ClassicalNetwork(n_actions, n_features, n_lstm_features, N_L1, N_lstm, batch_size, seed)

        self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.memory = []
        self.max_memory_size = memory_size
        self.memory_counter = 0

        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_alpha = 0.001 

        self.reward_store = []
        self.action_store = []
        self.delay_store = []
        self.energy_store = []
        self.epsilon_store = []

        self.lstm_history = deque(
            [torch.zeros(self.n_lstm_state).to(self.device) for _ in range(n_lstm_step)],
            maxlen=n_lstm_step
        )

    # --- FIX 1: Explicit LSTM Reset ---
    def reset_lstm(self):
        """Must be called at start of every episode to prevent history bleeding."""
        self.lstm_history.clear()
        for _ in range(self.n_lstm_step):
            self.lstm_history.append(torch.zeros(self.n_lstm_state).to(self.device))

    # --- FIX 3: Explicit Epsilon Decay ---
    def decay_epsilon(self):
        """Call this once per episode for stable decay."""
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        self.epsilon_store.append(self.epsilon)

    def choose_action(self, observation, inference=False):
        obs = np.asarray(observation, dtype=np.float32)[None, :]
        lstm_obs = torch.stack(list(self.lstm_history)).unsqueeze(0)

        if not inference and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        with torch.no_grad():
            q = self.eval_net(obs, lstm_obs)
            return int(torch.argmax(q[0]))

    def update_lstm(self, lstm_s):
        if isinstance(lstm_s, np.ndarray): t = torch.from_numpy(lstm_s).float().to(self.device)
        else: t = lstm_s.clone().detach().to(self.device)
        self.lstm_history.append(t)

    def store_transition(self, s, lstm_s, a, r, s_, lstm_s_, done=False):
        data = dict(s=s, lstm=lstm_s, a=a, r=r, s_=s_, lstm_=lstm_s_, done=done)
        if len(self.memory) < self.max_memory_size: self.memory.append(data)
        else: self.memory[self.memory_counter % self.max_memory_size] = data
        self.memory_counter += 1

    def learn(self):
        if len(self.memory) < self.batch_size + self.n_lstm_step: return
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        valid_idxs = []
        attempts = 0
        max_attempts = self.batch_size * 5
        while len(valid_idxs) < self.batch_size and attempts < max_attempts:
            idx = np.random.randint(0, len(self.memory) - self.n_lstm_step)
            window = self.memory[idx : idx + self.n_lstm_step]
            # Check for episode boundary crossings
            if not any(step['done'] for step in window[:-1]):
                valid_idxs.append(idx)
            attempts += 1
        if len(valid_idxs) < self.batch_size: return 

        s, s_, a, r, lstm_c, lstm_n = [], [], [], [], [], []
        for i in valid_idxs:
            window = self.memory[i : i + self.n_lstm_step]
            curr = window[-1]
            s.append(curr["s"])
            s_.append(curr["s_"])
            a.append(curr["a"])
            r.append(curr["r"])
            hist = [step["lstm"] for step in window]
            lstm_c.append(hist)
            lstm_n.append(hist[1:] + [curr["lstm_"]])

        s = torch.tensor(np.array(s), dtype=torch.float32).to(self.device)
        s_ = torch.tensor(np.array(s_), dtype=torch.float32).to(self.device)
        a = torch.tensor(np.array(a), dtype=torch.long).unsqueeze(1).to(self.device)
        r = torch.tensor(np.array(r), dtype=torch.float32).unsqueeze(1).to(self.device)
        lstm_c = torch.tensor(np.array(lstm_c), dtype=torch.float32).to(self.device)
        lstm_n = torch.tensor(np.array(lstm_n), dtype=torch.float32).to(self.device)

        batch_mean, batch_std = r.mean().item(), r.std().item()
        self.reward_mean = (1 - self.reward_alpha) * self.reward_mean + self.reward_alpha * batch_mean
        self.reward_std = (1 - self.reward_alpha) * self.reward_std + self.reward_alpha * batch_std
        r_norm = (r - self.reward_mean) / (max(self.reward_std, 0.1) + 1e-6)

        q_eval = self.eval_net(s, lstm_c).gather(1, a)
        with torch.no_grad():
            q_next = self.target_net(s_, lstm_n)
            q_target = r_norm + self.gamma * q_next.max(1, keepdim=True)[0]

        loss = self.loss_fn(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        if self.hybrid and self.learn_step_counter % 100 == 0:
            if self.eval_net.weights.grad is not None and torch.norm(self.eval_net.weights.grad).item() < 1e-4:
                print(f"[Warning] Barren Plateau detected.")
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), 1.0)
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % 100 == 0:
            print(f"[DQN] step={self.learn_step_counter}, eps={self.epsilon:.4f}, loss={loss.item():.4f}")
        with torch.no_grad():
            q_all = self.eval_net(s, lstm_c)
            self.q_mean_store.append(q_all.mean().item())
            self.q_std_store.append(q_all.std().item())

        # Gradient norm
        total_norm = 0.0
        for p in self.eval_net.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.grad_norm_store.append(total_norm)

        self.loss_store.append(loss.item())
        self.buffer_size_store.append(len(self.memory))


    def do_store_reward(self, ep, t, r):
        while ep >= len(self.reward_store): self.reward_store.append(np.zeros(self.n_time))
        self.reward_store[ep][t] = r

    def do_store_action(self, ep, t, a):
        while ep >= len(self.action_store): self.action_store.append(-np.ones(self.n_time))
        self.action_store[ep][t] = a

    def save(self, path=None):
        path = path or self.training_dir
        torch.save(self.eval_net.state_dict(), f"{path}/eval_net.pt")
        torch.save(self.target_net.state_dict(), f"{path}/target_net.pt")
        if self.hybrid: np.save(f"{path}/pqc_weights.npy", self.eval_net.weights.detach().cpu().numpy())

    def load(self, path):
        self.eval_net.load_state_dict(torch.load(f"{path}/eval_net.pt"))
        if os.path.exists(f"{path}/target_net.pt"): self.target_net.load_state_dict(torch.load(f"{path}/target_net.pt"))
        else: self.target_net.load_state_dict(self.eval_net.state_dict())
        if self.hybrid and os.path.exists(f"{path}/pqc_weights.npy"):
            self.eval_net.weights.data.copy_(torch.from_numpy(np.load(f"{path}/pqc_weights.npy")))