# metrics.py
import numpy as np
from scipy.stats import entropy, iqr

EPS = 1e-12

def action_entropy(actions, n_actions):
    counts = np.bincount(actions, minlength=n_actions) + EPS
    p = counts / counts.sum()
    return entropy(p)

def per_ue_action_entropy(action_matrix, n_actions):
    ent = []
    for ue in range(action_matrix.shape[1]):
        ent.append(action_entropy(action_matrix[:, ue], n_actions))
    return np.mean(ent), iqr(ent)

def action_kl_drift(prev_actions, curr_actions, n_actions):
    def dist(a):
        c = np.bincount(a, minlength=n_actions) + EPS
        return c / c.sum()
    return entropy(dist(prev_actions), dist(curr_actions))

def action_switching_rate(action_matrix):
    switches = action_matrix[1:] != action_matrix[:-1]
    return switches.mean()

def action_occupancy_variance(action_matrix, n_actions):
    T = action_matrix.shape[0]
    occ = []
    for t in range(T):
        occ.append(np.bincount(action_matrix[t], minlength=n_actions))
    occ = np.array(occ)
    return np.var(occ, axis=0).mean()
def jains_index(x):
    x = np.array(x) + EPS
    return (x.sum() ** 2) / (len(x) * (x ** 2).sum())

def action_jains_fairness(action_matrix):
    ue_modes = [np.bincount(action_matrix[:, ue]).max()
                for ue in range(action_matrix.shape[1])]
    return jains_index(ue_modes)
def action_entropy_iqr(entropy_list):
    return iqr(entropy_list)
def conditioned_action_kl(actions, states, bins, n_actions):
    # states = load or queue length per timestep
    out = []
    for b in bins:
        idx = np.where((states >= b[0]) & (states < b[1]))[0]
        if len(idx) < 5:
            continue
        a = actions[idx].flatten()
        out.append(action_entropy(a, n_actions))
    return np.var(out)
def action_sensitivity(loads, entropies):
    return np.polyfit(loads, entropies, 1)[0]
def latency_stats(delays):
    return {
        "mean": np.mean(delays),
        "p95": np.percentile(delays, 95),
        "p99": np.percentile(delays, 99),
        "iqr": iqr(delays),
        "max": np.max(delays)
    }
def latency_jains(per_ue_delays):
    means = [np.mean(d) for d in per_ue_delays if len(d) > 0]
    return jains_index(means)
def latency_cvar(delays, alpha=0.95):
    thresh = np.percentile(delays, alpha * 100)
    return delays[delays >= thresh].mean()
def worst_ue_latency(per_ue_delays):
    return max(np.mean(d) for d in per_ue_delays if len(d) > 0)
def energy_metrics(energies, per_ue_energies):
    return {
        "mean_energy": np.mean(energies),
        "energy_jain": jains_index([np.mean(e) for e in per_ue_energies]),
        "energy_iqr": iqr(energies)
    }
def reward_iqr(rewards):
    return iqr(rewards)
def policy_oscillation(action_switch_rates):
    return np.std(action_switch_rates)
def training_signal_variance(losses):
    return np.var(losses)
def action_diversity_gain(entropy_q, entropy_c):
    return entropy_q - entropy_c

