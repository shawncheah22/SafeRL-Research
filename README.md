# SafeRL-Research

**A research-focused environment for experimenting & developing Safe Reinforcement Learning (RL) algorithms.**

This repository is designed for **developing, validating, and benchmarking safe RL methods**, both standard and custom. It’s intended as a **playground for RL research** where safety constraints, hazard-aware learning, and novel algorithmic ideas can be tested.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Algorithms Implemented](#algorithms-implemented)
- [Safe RL Environments](#safe-rl-environments)
- [Experiments & Benchmarks](#experiments--benchmarks)

---

## Overview

SafeRL-Research is built to:

- Provide a **modular framework** for testing RL algorithms under safety constraints.
- Support both **standard RL algorithms** (DQN, PPO, SAC, etc.) and **custom innovations**.
- Facilitate **benchmarking and evaluation** in environments with hazards or safety-critical requirements.
- Serve as a **demonstration of advanced RL expertise** for recruiters, collaborators, and the research community.

### Technologies Used

- **Core Framework:** Python, OpenAI Safety Gymnasium
- **Libraries:** Matplotlib (for visualization)
- **Concepts:** Reinforcement Learning (RL), Safe RL, Markov Decision Processes (MDPs), Cost-Constrained RL, Hazard-Aware Learning

---

## Structure

The repository is structured to be highly modular and extensible:

- **`agents/base_agent.py`:** Defines the `BaseAgent` abstract class, ensuring all agents follow a consistent API for action selection, model updates, and saving/loading. This promotes a plug-and-play architecture for algorithm development.
- **`config/training_config.py`:** Centralizes key training hyperparameters (e.g., `NUM_EPISODES`, `MAX_STEPS_PER_EPISODE`), allowing for easy configuration of experiments without modifying the core logic.
- **`env.py`:** Wraps the `safety_gymnasium` environment to provide a consistent interface and handle environment-specific logic, such as collision detection.
- **`main.py`:** The main entry point for running experiments, orchestrating the agent-environment interaction loop.

---

## Key Features

- **Safe RL Focus:** Handle environments with penalties, hazards, or risk constraints.
- **Modular Algorithm Implementation:** Easily swap out policies, reward functions, and exploration strategies.
- **Custom Experiments:** Designed for testing novel RL ideas, risk-sensitive policies, or LLM-guided RL.
- **Visualization Tools:** Track performance, costs, and safety violations over time.
- **Benchmarking Support:** Compare performance across multiple algorithms and safety metrics.

---

## Algorithms Implemented

TBD

---

## Safe RL Environments

- Includes **OpenAI Safety Gymnasium / Safety Gym environments**
- Supports **goal-oriented, hazard-aware, and navigation tasks**

---

## Experiments & Benchmarks

This section will showcase the results of your experiments once you run them.

- **Performance Plots:** Reward curves, cost curves, etc.
- **Safety Analysis:** Number of safety violations, time-to-hazard plots.

---
