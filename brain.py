import tensorcircuit as tc
import tensorflow as tf
import numpy as np
import torch.nn as nn
import torch
from collections import deque


class QuantumNetwork(nn.Module):
<<<<<<< HEAD
    def __init__(
        self,
        n,              # number of qubits
        layers,         # number of PQC layers
        n_actions,
        n_lstm,
        n_lstm_state,
        n_features,
        n_l1,
        batch_size,
        seed,
        non_sequential,
        parallel_layers
    ):
        super(QuantumNetwork, self).__init__()

=======
    def __init__(self, n, layers, n_actions, n_lstm, n_lstm_state, n_features, n_l1, batch_size, dueling, seed, non_sequential, parallel_layers):
        super(QuantumNetwork, self).__init__()
>>>>>>> a4a0157 (first commit)
        self.n = n
        self.layers = layers
        self.parallel_layers = parallel_layers

<<<<<<< HEAD
        # TensorCircuit backend
        self.K = tc.set_backend("tensorflow")

        # LSTM part
        self.n_lstm = n_lstm
        self.batch_size = batch_size
        self.state_dim = n_lstm + n_features  # full classical state seen by PQC
=======
        self.K = tc.set_backend("tensorflow")
        self.c = tc.Circuit(n)

        self.n_lstm = n_lstm
        self.batch_size = batch_size
        self.dueling = dueling
>>>>>>> a4a0157 (first commit)

        torch.cuda.manual_seed(seed)

        self.lstm = nn.LSTM(n_lstm_state, n_lstm, batch_first=True)

<<<<<<< HEAD
        # Quantum parameters and layer
        if not non_sequential:
            # weights: [layers, n_qubits, 3] for RZ-RY-RZ
            self.weights = torch.nn.Parameter(torch.empty(layers, n, 3))
            self.quantum_layer = tc.interfaces.torch_interface(
                self.K.vmap(self.vqc, vectorized_argnums=0), jit=True
            )
        else:
            # we won't use this mode right now, but keep signature
            self.weights = torch.nn.Parameter(torch.empty(parallel_layers, layers, n, 3))
            self.quantum_layer = tc.interfaces.torch_interface(
                self.K.vmap(self.non_sequential_vqc, vectorized_argnums=0), jit=True
            )

        nn.init.xavier_uniform_(self.weights)

        # Classical head: takes [full_state, quantum_features]
        self.nn_output = nn.Sequential(
            nn.Linear(self.state_dim + self.n, n_l1),
            nn.ReLU()
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.output = nn.Linear(n_l1, n_actions)

        # Init all linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.3)
                nn.init.constant_(m.bias, 0.1)

    @tf.function
    def vqc(self, state, weights):
        """
        state  : 1D tensor of length D = self.state_dim (full [lstm_output, s])
        weights: [layers, n_qubits, 3]
        """
        # length of state vector D  (tf-style)
        D = self.K.shape_tensor(state)[0]

        # fresh circuit every call
        c = tc.Circuit(self.n)

        # optional: start in Hadamard basis
        for q in range(self.n):
            c.h(q)

        # data reuploading + param layers
        for layer in range(self.layers):
            for q in range(self.n):
                # which classical feature does this (layer, qubit) see?
                idx = (layer * self.n + q) % D
                angle = state[idx]

                # encode data
                c.rx(q, theta=angle)
                c.ry(q, theta=angle)

                # trainable rotations
                c.rz(q, theta=weights[layer, q, 0])
                c.ry(q, theta=weights[layer, q, 1])
                c.rz(q, theta=weights[layer, q, 2])

            # ring entanglement
            for q in range(self.n - 1):
                c.cnot(q, q + 1)
            c.cnot(self.n - 1, 0)

        # measure Z on each qubit → feature vector of length n
        output = self.K.stack(
            [self.K.real(c.expectation((tc.gates.z(), [q]))) for q in range(self.n)]
        )
=======

        if not non_sequential:
            self.weights = torch.nn.Parameter(torch.empty(layers, n, 3))
            self.quantum_layer = tc.interfaces.torch_interface(self.K.vmap(self.vqc, vectorized_argnums=0), jit=True)

            self.nn_input = nn.Sequential(
                    nn.Linear(n_lstm + n_features, n),
                    nn.ReLU()
                    )
        else:
            self.weights = torch.nn.Parameter(torch.empty(parallel_layers, layers, n, 3))
            self.quantum_layer = tc.interfaces.torch_interface(self.K.vmap(self.non_sequential_vqc, vectorized_argnums=0), jit=True)

            self.nn_input = nn.Sequential(
                    nn.Linear(n_lstm + n_features, n // self.parallel_layers),
                    nn.ReLU()
                    )

        nn.init.xavier_uniform_(self.weights)

        self.nn_output = nn.Sequential(
                nn.Linear(n, n_l1),
                nn.ReLU()
                )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if dueling:
            self.value_stream = nn.Linear(n_l1, 1)
            self.advantage_stream = nn.Linear(n_l1, n_actions)
        else:
            self.output = nn.Linear(n_l1, n_actions)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=0.3)
                nn.init.constant_(m.bias, 0.1)

    @tf.function
    def vqc(self, inputs, weights):
        for i in range(self.layers):
            for j in range(self.n):
                self.c.rx(j, theta=inputs[j])
                self.c.ry(j, theta=inputs[j])
            for j in range(self.n):
                self.c.rz(j, theta=weights[i, j, 0])
                self.c.ry(j, theta=weights[i, j, 1])
                self.c.rz(j, theta=weights[i, j, 2])
            for j in range(self.n - 1):
                self.c.cnot(j, j + 1)
            self.c.cnot(self.n - 1, 0)

        output = self.K.stack([self.K.real(self.c.expectation((tc.gates.z(), [i]))) for i in range(self.n)])
>>>>>>> a4a0157 (first commit)

        return output

    @tf.function
    def non_sequential_vqc(self, inputs, weights):
<<<<<<< HEAD
        # keep the old non_sequential implementation if you need it,
        # or raise an error if you don't plan to use this mode:
        raise NotImplementedError("non_sequential_vqc is not implemented with the new architecture.")

    def draw_circuit(self, file):
        """
        Optional: you can keep a simple draw using random inputs if you want.
        Not used in training if the call in HybridDQN.__init__ is commented out.
        """
        inputs = tf.random.uniform((self.state_dim,))
        # simple dummy weights for drawing
        weights = tf.random.uniform((self.layers, self.n, 3))

        c = tc.Circuit(self.n)

        D = self.state_dim
        for layer in range(self.layers):
            for q in range(self.n):
                idx = (layer * self.n + q) % D
                angle = inputs[idx]
                c.h(q)
                c.rx(q, theta=angle)
                c.ry(q, theta=angle)
            for q in range(self.n - 1):
                c.cnot(q, q + 1)
            c.cnot(self.n - 1, 0)

        fig = c.draw(output="mpl")
        fig.set_size_inches(max(16, self.layers * 3), self.n)
        fig.savefig(file, bbox_inches="tight")

    def forward(self, s, lstm_s):
        # device placement + numpy → tensor conversion
=======
        results = []
        for i in range(self.parallel_layers):
            for j in range(self.layers):
                for k in range(self.n // self.parallel_layers):
                    self.c.rx(self.parallel_layers * i + k, theta=inputs[k])
                for k in range(self.n // self.parallel_layers):
                    self.c.rz(self.parallel_layers * i + k, theta=weights[i, j, k, 0])
                    self.c.ry(self.parallel_layers * i + k, theta=weights[i, j, k, 1])
                    self.c.rz(self.parallel_layers * i + k, theta=weights[i, j, k, 2])
                for k in range(self.n // self.parallel_layers - 1):
                    self.c.cnot(self.parallel_layers * i + k, self.parallel_layers * i + k + 1)
                self.c.cnot(self.n // self.parallel_layers - 1, 0)

            result = self.K.stack([self.K.real(self.c.expectation((tc.gates.z(), [j]))) for j in range(self.parallel_layers * i, self.parallel_layers * i + self.n // self.parallel_layers)])
            results.append(result)
        output = self.K.shape_concat(results, axis=0)
        return output

    def forward(self, s, lstm_s):
>>>>>>> a4a0157 (first commit)
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s).to(self.device)
        if isinstance(lstm_s, np.ndarray):
            lstm_s = torch.FloatTensor(lstm_s).to(self.device)

<<<<<<< HEAD
        # LSTM over history [B, T, n_lstm_state]
        h_0 = torch.zeros(1, lstm_s.shape[0], self.n_lstm, device=lstm_s.device)
        c_0 = torch.zeros(1, lstm_s.shape[0], self.n_lstm, device=lstm_s.device)
        lstm_output, _ = self.lstm(lstm_s, (h_0, c_0))
        lstm_output = lstm_output[:, -1, :].reshape(-1, self.n_lstm)  # [B, n_lstm]

        # full classical state seen by PQC
        state_full = torch.cat([lstm_output, s], dim=1)               # [B, D]

        # bound angles to [-π, π] for stability
        state_norm = torch.tanh(state_full) * np.pi                   # [B, D]

        # PQC with full-state data reuploading
        # quantum_layer is vmapped over batch dim, so this is fine:
        q_features = self.quantum_layer(state_norm, self.weights)     # [B, n_qubits]

        # merge classical and quantum features
        merged = torch.cat([state_full, q_features], dim=1)           # [B, D + n_qubits]

        x = self.nn_output(merged)
        return self.output(x)

class ClassicalNetwork(nn.Module):
    def __init__(self, n_actions, n_features, n_lstm_state, n_l1, n_lstm, batch_size, seed):
=======
        h_0 = torch.zeros(1, lstm_s.shape[0], self.n_lstm, device=lstm_s.device)
        c_0 = torch.zeros(1, lstm_s.shape[0], self.n_lstm, device=lstm_s.device)

        lstm_output, _ = self.lstm(lstm_s, (h_0, c_0))
        lstm_output = lstm_output[:, -1, :].reshape(-1, self.n_lstm)

        x = torch.cat([lstm_output, s], dim=1)

        x = self.nn_input(x)
        x = self.quantum_layer(x, self.weights)
        x = self.nn_output(x)

        if self.dueling:
            V = self.value_stream(x)
            A = self.advantage_stream(x)
            out = V + A - A.mean(dim=1, keepdim=True)
            return out
        else:
            return self.output(x)

class ClassicalNetwork(nn.Module):
    def __init__(self, n_actions, n_features, n_lstm_state, n_l1, n_lstm, batch_size, dueling, seed):
>>>>>>> a4a0157 (first commit)
        super(ClassicalNetwork, self).__init__()

        self.n_lstm = n_lstm
        self.batch_size = batch_size
<<<<<<< HEAD
=======
        self.dueling = dueling
>>>>>>> a4a0157 (first commit)

        torch.cuda.manual_seed(seed)

        self.lstm = nn.LSTM(n_lstm_state, n_lstm, batch_first=True)

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(n_lstm + n_features, n_l1)
        self.fc12 = nn.Linear(n_l1, n_l1)

<<<<<<< HEAD
        self.output = nn.Linear(n_l1, n_actions)
=======
        if dueling:
            self.value_stream = nn.Linear(n_l1, 1)
            self.advantage_stream = nn.Linear(n_l1, n_actions)
        else:
            self.output = nn.Linear(n_l1, n_actions)
>>>>>>> a4a0157 (first commit)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=0.3)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, s, lstm_s):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s).to(device)
        if isinstance(lstm_s, np.ndarray):
            lstm_s = torch.FloatTensor(lstm_s).to(device)

        h_0 = torch.zeros(1, lstm_s.shape[0], self.n_lstm, device=lstm_s.device)
        c_0 = torch.zeros(1, lstm_s.shape[0], self.n_lstm, device=lstm_s.device)
        lstm_output, _ = self.lstm(lstm_s, (h_0, c_0))
        lstm_output = lstm_output[:, -1, :].reshape(-1, self.n_lstm)

        x = torch.cat([lstm_output, s], dim=1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc12(x))

<<<<<<< HEAD
        return self.output(x)
=======
        if self.dueling:
            V = self.value_stream(x)
            A = self.advantage_stream(x)
            out = V + A - A.mean(dim=1, keepdim=True)
            return out
        else:
            return self.output(x)
>>>>>>> a4a0157 (first commit)
class HybridDQN:

    def __init__(self,
                 n_actions,                  # the number of actions
                 n_features,
                 n_lstm_features,
                 n_time,
                 learning_rate=0.001,
                 reward_decay=0.9,
                 e_greedy=0.99,
                 replace_target_iter=200,  # each 200 steps, update target net
                 memory_size=500,  # maximum of memory
                 batch_size=32,
                 e_greedy_increment=0.00025,
                 n_lstm_step=10,
<<<<<<< HEAD
=======
                 dueling=False,
                 double_q=False,
>>>>>>> a4a0157 (first commit)
                 N_L1=20,
                 N_lstm=20,
                 optimizer='rms_prop',
                 seed=0,
                 hybrid=False,
                 qubits=3,
                 layers=3,
                 non_sequential=False,
<<<<<<< HEAD
                 parallel_layers=2,
                 training_dir=''):
=======
                 parallel_layers=2):
>>>>>>> a4a0157 (first commit)

        self.n_actions = n_actions
        self.n_features = n_features
        self.n_time = n_time
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size    # select self.batch_size number of time sequence for learning
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
<<<<<<< HEAD
=======
        self.dueling = dueling
        self.double_q = double_q
>>>>>>> a4a0157 (first commit)
        self.learn_step_counter = 0
        self.N_L1 = N_L1

        # lstm
        self.N_lstm = N_lstm
        self.n_lstm_step = n_lstm_step       # step_size in lstm
        self.n_lstm_state = n_lstm_features  # [fog1, fog2, ...., fogn, M_n(t)]

        if optimizer not in ['adam', 'rms_prop', 'gd']:
            raise SystemExit("Invalid optimizer: {optimizer}.\nChoose one of " +
                             "['adam', 'rms_prop', 'gd'], via CLI with flag --optimizer")


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        if hybrid:
<<<<<<< HEAD
            self.target_net = QuantumNetwork(qubits, layers, n_actions, N_lstm, self.n_lstm_state, n_features, N_L1, batch_size, seed, non_sequential, parallel_layers)
            #self.target_net.draw_circuit(training_dir + 'circuit.png')
            self.target_net.to(self.device)
            self.eval_net = QuantumNetwork(qubits, layers, n_actions, N_lstm, self.n_lstm_state, n_features, N_L1, batch_size, seed, non_sequential, parallel_layers)
            self.eval_net.to(self.device)
        else:
            self.target_net = ClassicalNetwork(n_actions, n_features, self.n_lstm_state, N_L1, N_lstm, batch_size, seed)
            self.target_net.to(self.device)
            self.eval_net = ClassicalNetwork(n_actions, n_features, self.n_lstm_state, N_L1, N_lstm, batch_size, seed)
=======
            self.target_net = QuantumNetwork(qubits, layers, n_actions, N_lstm, self.n_lstm_state, n_features, N_L1, batch_size, dueling, seed, non_sequential, parallel_layers)
            self.target_net.to(self.device)
            self.eval_net = QuantumNetwork(qubits, layers, n_actions, N_lstm, self.n_lstm_state, n_features, N_L1, batch_size, dueling, seed, non_sequential, parallel_layers)
            self.eval_net.to(self.device)
        else:
            self.target_net = ClassicalNetwork(n_actions, n_features, self.n_lstm_state, N_L1, N_lstm, batch_size, dueling, seed)
            self.target_net.to(self.device)
            self.eval_net = ClassicalNetwork(n_actions, n_features, self.n_lstm_state, N_L1, N_lstm, batch_size, dueling, seed)
>>>>>>> a4a0157 (first commit)
            self.eval_net.to(self.device)

        if optimizer == 'adam':
            self.eval_optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        elif optimizer == 'gd':
            self.eval_optimizer = torch.optim.SGD(self.eval_net.parameters(), lr=learning_rate)
        elif optimizer == 'rms_prop':
            self.eval_optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=learning_rate)

        self.loss = nn.MSELoss()

        # initialize zero memory np.hstack((s, [a, r], s_, lstm_s, lstm_s_))
        self.memory = torch.zeros((self.memory_size, self.n_features + 1 + 1
                                + self.n_features + self.n_lstm_state + self.n_lstm_state)).to(self.device)
        self.memory_counter = 0

        self.reward_store = list()
        self.action_store = list()
        self.delay_store = list()
        self.energy_store = list()

        self.lstm_history = deque(maxlen=self.n_lstm_step)
        for _ in range(self.n_lstm_step):
            self.lstm_history.append(torch.zeros(self.n_lstm_state).to(self.device))

        self.store_q_value = list()

    def store_transition(self, s, lstm_s,  a, r, s_, lstm_s_):
        # RL.store_transition(observation,action,reward,observation_)
        # hasattr(object, name), if object has name attribute
        # store np.hstack((s, [a, r], s_, lstm_s, lstm_s_))
        transition = np.hstack((s, [a, r], s_, lstm_s, lstm_s_))  # stack in horizontal direction

        # if memory overflows, replace old memory with new one
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = torch.from_numpy(transition).to(self.device)
        self.memory_counter += 1

    def update_lstm(self, lstm_s):
        if isinstance(lstm_s, np.ndarray):
            lstm_s = torch.from_numpy(lstm_s).to(self.device)

        self.lstm_history.append(lstm_s)

    def choose_action(self, observation, inference=False):
        observation = observation[np.newaxis, :]

        if inference or np.random.uniform() < self.epsilon:

            # lstm only contains history, there is no current observation
            lstm_observation = torch.cat([t.unsqueeze(0) for t in self.lstm_history], dim=0).to(self.device).float()

            with torch.no_grad():
                actions_value = self.eval_net(observation, lstm_observation.reshape(1, self.n_lstm_step, self.n_lstm_state))

            self.store_q_value.append({'observation': observation, 'q_value': actions_value})

            action = torch.argmax(actions_value)

        else:

            action = np.random.randint(0, self.n_actions)

        return action

    def update_target_net(self, target_net, eval_net):
        target_net.load_state_dict(eval_net.state_dict())

    def learn(self):

        # check if replace target_net parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.update_target_net(self.target_net, self.eval_net)

        # randomly pick [batch_size] memory from memory np.hstack((s, [a, r], s_, lstm_s, lstm_s_))
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size - self.n_lstm_step, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter - self.n_lstm_step, size=self.batch_size)\

        batch_memory = self.memory[sample_index, :self.n_features+1+1+self.n_features]
        lstm_batch_memory = torch.zeros(self.batch_size, self.n_lstm_step, self.n_lstm_state * 2).to(self.device)
        for ii in range(len(sample_index)):
            for jj in range(self.n_lstm_step):
                lstm_batch_memory[ii,jj,:] = self.memory[sample_index[ii]+jj,
                                              self.n_features+1+1+self.n_features:]

        next_state = batch_memory[:, -self.n_features:]
        lstm_next_state = lstm_batch_memory[:,:,self.n_lstm_state:]
        current_state = batch_memory[:, :self.n_features]
        lstm_current_state = lstm_batch_memory[:,:,:self.n_lstm_state]
        reward = batch_memory[:, self.n_features + 1]
        action = batch_memory[:, self.n_features].to(torch.int64)

        q_eval = self.eval_net(current_state, lstm_current_state)
        q_target = q_eval.clone().detach()

        with torch.no_grad():
            q_target_next = self.target_net(next_state, lstm_next_state)

<<<<<<< HEAD
            q_next_selected = q_target_next.max(dim=1)[0]
=======
            if self.double_q:
                q_eval_next = self.eval_net(next_state, lstm_next_state)
                max_action = q_eval_next.argmax(dim=1)
                q_next_selected = q_target_next.gather(1, max_action.unsqueeze(1)).squeeze(1)
            else:
                q_next_selected = q_target_next.max(dim=1)[0]
>>>>>>> a4a0157 (first commit)

            q_target_values = reward + self.gamma * q_next_selected

            batch_indices = torch.arange(self.batch_size, dtype=torch.int32).to(self.device)
            q_target[batch_indices, action] = q_target_values

        loss = self.loss(q_target, q_eval)

        print("Current Loss: ", loss)

        self.eval_optimizer.zero_grad()
        loss.backward()
        self.eval_optimizer.step()

        # gradually increase epsilon
        self.epsilon = min(self.epsilon + self.epsilon_increment, self.epsilon_max)
        self.learn_step_counter += 1

    def do_store_reward(self, episode, time, reward):
        while episode >= len(self.reward_store):
            self.reward_store.append(torch.zeros(self.n_time).to(self.device))
        self.reward_store[episode][time] = reward

    def do_store_action(self,episode,time, action):
        while episode >= len(self.action_store):
            self.action_store.append(- torch.ones(self.n_time).to(self.device))
        self.action_store[episode][time] = action

    def do_store_delay(self, episode, time, delay):
        while episode >= len(self.delay_store):
            self.delay_store.append(np.zeros([self.n_time]))
        self.delay_store[episode][time] = delay

    def do_store_energy(self, episode, time, energy):
        while episode >= len(self.energy_store):
            self.energy_store.append(np.zeros([self.n_time]))
        self.energy_store[episode][time] = energy
