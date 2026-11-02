# DDoS Attack‑Defense Game

## Project Overview
This is a small academic mini‑project that demonstrates how **game theory** can model the strategic interaction between a DDoS attacker and a network defender. The app builds a 2×2 payoff matrix from user‑provided parameters, computes best‑responses, finds pure‑ and mixed‑strategy Nash equilibria, runs a simple repeated‑round simulation, and visualizes the results.

## Installation
```bash
# (Optional) Create a virtual environment
python -m venv venv
# Activate the environment
# Windows:
venv\\Scripts\\activate
# Unix/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the App
```bash
streamlit run app.py
```
Open the displayed URL in a browser. Use the sidebar to change parameters, view the payoff matrix, examine equilibrium analysis, and run the simulation.

## Files
- `app.py` – Streamlit UI
- `game.py` – Core calculations (payoff matrix, best‑responses, Nash equilibria, simulation)
- `requirements.txt` – Python dependencies
- `README.md` – This documentation

## Academic Context
The model captures the trade‑off between **attack impact** (damage) and **costs** of launching attacks or deploying defenses. By analysing Nash equilibria we discuss optimal strategic choices for both players.
