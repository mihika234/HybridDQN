"""
train.py - FINAL CORRECTED VERSION
"""

import os
import time
import json
import random
import argparse
from datetime import datetime
from shutil import rmtree

import numpy as np
import matplotlib.pyplot as plt

from fog_env import Offload
from brain import HybridDQN
from utils import plot_graphs

np.set_printoptions(threshold=np.inf)


def safe_mean(xs):
    xs = np.array(xs, dtype=np.float64)
    xs = xs[~np.isnan(xs)]
    return float(xs.mean()) if xs.size > 0 else float("nan")


def generate_bitarrive(env):
    """Generate task arrivals for one episode."""
    bitarrive = np.random.uniform(
        env.min_bit_arrive, env.max_bit_arrive, size=(env.n_time, env.n_iot)
    )
    bitarrive *= (np.random.rand(env.n_time, env.n_iot) < env.task_arrive_prob)
    bitarrive[-env.max_delay :, :] = 0.0
    return bitarrive


def CTDE_train(
    env,
    central_server,
    iot_RL_list,
    num_episodes,
    learning_freq=10,
    show=False,
    random_policy=False,
    training_dir=None,
):
    start_time = time.time()
    RL_step = 0

    episode_rewards, episode_dropped = [], []
    episode_delay, episode_energy = [], []

    fig, axs = plt.subplots(4, figsize=(10, 12), sharex=True)
    
    diagnostics = {
    "reward": [],
    "drop": [],
    "delay": [],
    "energy": [],
    "epsilon": [],
    "episode_time": [],
    "fog_drop": [],
    "trans_drop": [],
    "iot_drop": [],
}
    action_hist = np.zeros(env.n_actions, dtype=np.int64)


    for episode in range(num_episodes):

        # ---------------------------
        # 1. Reset Env & LSTMs
        # ---------------------------
        ep_start_time = time.time()

        bitarrive = generate_bitarrive(env)
        observation_all, lstm_state_all = env.reset(bitarrive)
        
        # Reset Central Brain
        central_server.reset_lstm()
        
        # FIX 2: Reset IoT Agents (Logging) to prevent memory leak across episodes
        for iot in iot_RL_list:
            iot.reset_lstm()

        # ---------------------------
        # History buffer
        # ---------------------------
        history = [
            [
                {
                    "obs": np.zeros(env.n_features),
                    "lstm": np.zeros(env.n_lstm_state),
                    "act": 0,
                    "obs_": np.zeros(env.n_features),
                    "lstm_": np.zeros(env.n_lstm_state),
                }
                for _ in range(env.n_iot)
            ]
            for _ in range(env.n_time)
        ]

        ep_rewards, ep_drops = [], []
        ep_delays, ep_energies = [], []

        # ---------------------------
        # Episode loop
        # ---------------------------
        step_diag = {
    "fog_congestion": [],
    "finished_tasks": [],
    "dropped_tasks": [],
}

        while True:

            action_all = np.zeros(env.n_iot, dtype=int)

            for i in range(env.n_iot):
                obs = observation_all[i]
                if np.sum(obs[(2 + 2 * env.n_fog):]) == 0:
                    action_all[i] = 0 # Safe No-Op
                else:
                    if random_policy:
                        action_all[i] = np.random.randint(env.n_actions)
                    else:
                        action_all[i] = central_server.choose_action(obs)
            for a in action_all:
                action_hist[a] += 1

                # Log action (Minor Issue 3 accepted: logging 0s is fine for timeline completeness)
                iot_RL_list[i].do_store_action(episode, env.time_count, action_all[i])

            obs_, lstm_, done, info = env.step(action_all)
            step_diag["fog_congestion"].append(env.fog_iot_m.mean())
            step_diag["finished_tasks"].append(len(info["finished"]))
            step_diag["dropped_tasks"].append(
                sum(1 for e in info["finished"] if e["dropped"])
            )


            # FIX 1: Correct CTDE Aggregation
            # Instead of feeding just the first IoT's view, we average the global state view.
            # In your specific env, rows are identical, but this is the mathematically correct operation for CTDE.
            central_server.update_lstm(np.mean(lstm_, axis=0))

            # Save context
            t_now = max(0, env.time_count - 1)
            if t_now < env.n_time:
                for i in range(env.n_iot):
                    history[t_now][i]["obs"] = observation_all[i].copy()
                    history[t_now][i]["lstm"] = lstm_state_all[i].copy()
                    history[t_now][i]["act"] = action_all[i]
                    history[t_now][i]["obs_"] = obs_[i].copy()
                    history[t_now][i]["lstm_"] = lstm_[i].copy()

            # Process Finished Tasks
            # Process Finished Tasks
            for evt in info["finished"]:
                i = evt["iot"]
                t0 = int(evt["start_time"])
                r = evt["reward"]
                dropped = evt["dropped"]

                ep_rewards.append(r)
                ep_drops.append(1 if dropped else 0)

                if not dropped:
                    ep_delays.append(evt.get("delay", np.nan))
                    ep_energies.append(evt.get("energy", np.nan))

                h = history[t0][i]
                
                # OLD BROKEN LOGIC:
                # is_terminal = dropped 
                
                # NEW FIXED LOGIC:
                # Mark as terminal if the task failed (dropped) OR if the episode ended (done).
                # This prevents the LSTM from training on invisible boundaries between episodes.
                is_terminal = dropped or done

                central_server.store_transition(
                    h["obs"], h["lstm"], h["act"],
                    r,
                    h["obs_"], h["lstm_"],
                    done=is_terminal
                )

                iot_RL_list[i].do_store_reward(episode, t0, r)
            # Learning
            RL_step += 1
            if RL_step > 200 and RL_step % learning_freq == 0:
                central_server.learn()

            observation_all = obs_
            lstm_state_all = lstm_

            if done:
                break
        
        # Decay Epsilon per episode
        central_server.decay_epsilon()

        # ---------------------------
        # Stats & Plotting
        # ---------------------------
        episode_rewards.append(safe_mean(ep_rewards))
        episode_dropped.append(safe_mean(ep_drops))
        episode_delay.append(safe_mean(ep_delays))
        episode_energy.append(safe_mean(ep_energies))

        print(
            f"Ep {episode:4d} | "
            f"R: {episode_rewards[-1]:7.3f} | "
            f"Drop: {episode_dropped[-1]:5.3f} | "
            f"Dly: {episode_delay[-1]:5.3f} | "
            f"Eps: {central_server.epsilon:.3f}"
        )
        diagnostics["reward"].append(episode_rewards[-1])
        diagnostics["drop"].append(episode_dropped[-1])
        diagnostics["delay"].append(episode_delay[-1])
        diagnostics["energy"].append(episode_energy[-1])
        diagnostics["epsilon"].append(central_server.epsilon)
        diagnostics["episode_time"].append(time.time() - ep_start_time)

        diagnostics["fog_drop"].append(env.drop_fog_count)
        diagnostics["trans_drop"].append(env.drop_trans_count)
        diagnostics["iot_drop"].append(env.drop_iot_count)


        if episode % 10 == 0:
            plot_graphs(
                axs, episode_rewards, episode_dropped, episode_delay, episode_energy,
                show=show, save=True, path=training_dir,
            )

    plot_graphs(
        axs, episode_rewards, episode_dropped, episode_delay, episode_energy,
        show=show, save=True, path=training_dir,
    )
    # Normalize and save action histogram
    action_hist = action_hist.astype(np.float64)
    action_hist /= max(action_hist.sum(), 1)

    np.save(os.path.join(training_dir, "results", "action_hist.npy"), action_hist)
    np.save(training_dir + "/results/episode_metrics.npy", diagnostics)

    np.save(training_dir + "/results/loss.npy",
            np.array(central_server.loss_store))

    np.save(training_dir + "/results/q_stats.npy",
            np.vstack([
                central_server.q_mean_store,
                central_server.q_std_store
            ]))

    np.save(training_dir + "/results/grad_norm.npy",
            np.array(central_server.grad_norm_store))

    np.save(training_dir + "/results/buffer_size.npy",
            np.array(central_server.buffer_size_store))
    import pandas as pd
    pd.DataFrame(diagnostics).to_csv(
        training_dir + "/results/episode_metrics.csv",
        index=False
    )

    print(f"Training finished in {time.time() - start_time:.2f}s")


def evaluate(env, central_server, num_episodes, random_policy=False):
    rewards, drops = [], []
    delays, energies = [], []

    for ep in range(num_episodes):
        print(f"[EVAL] episode {ep}")
        bitarrive = generate_bitarrive(env)
        obs, lstm = env.reset(bitarrive)
        
        # Reset LSTM for clean evaluation
        central_server.reset_lstm()
        
        while True:
            action_all = np.zeros(env.n_iot, dtype=int)
            for i in range(env.n_iot):
                o = obs[i]
                if np.sum(o[(2 + 2 * env.n_fog):]) == 0:
                    action_all[i] = 0
                else:
                    action_all[i] = (
                        np.random.randint(env.n_actions)
                        if random_policy
                        else central_server.choose_action(o, inference=True)
                    )

            obs, lstm, done, info = env.step(action_all)
            
            # Correct aggregation during eval too
            central_server.update_lstm(np.mean(lstm, axis=0))

            for evt in info["finished"]:
                rewards.append(evt["reward"])
                drops.append(1 if evt["dropped"] else 0)
                if not evt["dropped"]:
                    delays.append(evt.get("delay", np.nan))
                    energies.append(evt.get("energy", np.nan))

            if done: break

    return {
        "avg_rewards": safe_mean(rewards),
        "avg_dropped": safe_mean(drops),
        "avg_delay": safe_mean(delays),
        "avg_energy": safe_mean(energies)
    }


def main(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    # torch seed handled in brain.py

    now = datetime.now()
    training_dir = args.path or f"training/{now:%Y-%m-%d_%H-%M-%S}/"

    if os.path.exists(training_dir): rmtree(training_dir)
    os.makedirs(training_dir + "/plots", exist_ok=True)
    os.makedirs(training_dir + "/results", exist_ok=True)
    os.makedirs(training_dir + "/params", exist_ok=True)

    with open(training_dir + "/params/params.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    env = Offload(
        args.num_iot, args.num_fog,
        args.num_time + args.max_delay,
        args.max_delay, args.task_arrival_prob,
    )

    central_server = HybridDQN(
        env.n_actions, env.n_features, env.n_lstm_state, env.n_time,
        learning_rate=args.lr, reward_decay=0.9, e_greedy=0.99,
        replace_target_iter=200, memory_size=args.memory_size,
        batch_size=args.batch_size, seed=args.seed,
        hybrid=args.hybrid, qubits=args.qubits, layers=args.layers,
        training_dir=training_dir,
    )

    # Placeholder for logging
    iot_RL_list = [
        HybridDQN(
            env.n_actions, env.n_features, env.n_lstm_state, env.n_time,
            learning_rate=args.lr, memory_size=10, batch_size=10,
            hybrid=False, training_dir=training_dir,
        ) for _ in range(args.num_iot)
    ]

    CTDE_train(
        env, central_server, iot_RL_list,
        args.num_episodes, learning_freq=args.learning_freq,
        show=args.plot, random_policy=args.random,
        training_dir=training_dir,
    )

    print("Evaluation (50 eps):", evaluate(env, central_server, 50))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iot", type=int, default=50)
    parser.add_argument("--num_fog", type=int, default=5)
    parser.add_argument("--task_arrival_prob", type=float, default=0.1)
    parser.add_argument("--num_time", type=int, default=100)
    parser.add_argument("--max_delay", type=int, default=20)
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--learning_freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--qubits", type=int, default=3)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--memory_size", type=int, default=10000)

    main(parser.parse_args())