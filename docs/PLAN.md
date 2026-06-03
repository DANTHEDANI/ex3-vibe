# Architecture & Planning (PLAN)

## 1. C4 Model - Context
The system retrieves workout data from Kaggle, processes it into a Trajectory, trains a World Model (LSTM), and then uses it as an Environment to train RL agents (REINFORCE & A2C).

## 2. Component Diagram
- **Data Pipeline**: Downloads and filters Kaggle CSVs, computes rolling volumes.
- **World Model**: PyTorch LSTM that takes `(s_t, a_t)` and predicts `s_{t+1}`.
- **RL Agents**: Actor-Critic models taking states to predict actions to maximize rewards.

## 3. ADRs (Architecture Decision Records)
- **ADR 1**: Use `uv` for package management instead of `pip`.
- **ADR 2**: Implement an `API Gatekeeper` (SDK) to route all requests to models and data pipelines.
- **ADR 3**: Use `torch` for LSTM and RL implementations due to ease of defining custom network architectures and autograd.
