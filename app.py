import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from game import (
    build_payoff_matrices,
    best_responses,
    pure_nash,
    mixed_nash,
    simulate_game,
)

# ---------------------------------------------------------------------------
# Page configuration & custom CSS for a dark cyber‑security theme
# ---------------------------------------------------------------------------
st.set_page_config(page_title="DDoS Attack‑Defense Game", layout="wide")

# Apply a simple dark theme using markdown + CSS (Streamlit does not expose direct theme API)
dark_css = """
<style>
    .main {background-color: #0b0e13; color: #e0e0e0;}
    .sidebar .sidebar-content {background-color: #14181f;}
    .card {background-color: #1e222a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;}
    .metric-card {background-color: #1e222a; padding: 1rem; border-radius: 8px; text-align: center;}
    .metric {font-size: 2rem; font-weight: bold; color: #00ffab;}
    .metric-label {font-size: 1rem; color: #a0a0a0;}
    .highlight {background-color: #2a2f38; border-left: 4px solid #ff5733; padding-left: 0.5rem;}
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar – Game parameters (grouped for clarity)
# ---------------------------------------------------------------------------
st.sidebar.header("Game Parameters")

st.sidebar.subheader("Attacker")
attack_cost_low = st.sidebar.number_input("Low Attack Cost", min_value=0.0, value=1.0, help="Cost of launching a low‑impact attack")
attack_cost_high = st.sidebar.number_input("High Attack Cost", min_value=0.0, value=3.0, help="Cost of launching a high‑impact attack")

damage = st.sidebar.number_input("Damage (if attack succeeds)", min_value=0.0, value=5.0, help="Loss incurred by the defender when an attack is successful")

st.sidebar.subheader("Defender")
defense_cost_low = st.sidebar.number_input("Low Defense Cost", min_value=0.0, value=1.0, help="Cost of a minimal defence posture")

defense_cost_high = st.sidebar.number_input("High Defense Cost", min_value=0.0, value=3.0, help="Cost of a strong defence posture")

defence_effectiveness = st.sidebar.slider(
    "Defense Effectiveness",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    step=0.05,
    help="Probability that high defence blocks a high attack",
)

# ---------------------------------------------------------------------------
# Core calculations (payoff matrices, best responses, equilibria)
# ---------------------------------------------------------------------------
attacker_df, defender_df = build_payoff_matrices(
    attack_cost_low,
    attack_cost_high,
    defense_cost_low,
    defense_cost_high,
    damage,
    defence_effectiveness,
)

attacker_br, defender_br = best_responses(attacker_df, defender_df)
pure_eq = pure_nash(attacker_br, defender_br)
(attacker_mix, defender_mix) = mixed_nash(attacker_df, defender_df)

# ---------------------------------------------------------------------------
# Top header – title + subtitle
# ---------------------------------------------------------------------------
st.markdown("""<h1 style='text-align: center; color: #e0e0e0;'>DDoS ATTACK‑DEFENSE GAME</h1>
<h3 style='text-align: center; color: #a0a0a0;'>Game Theory Simulation</h3>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Metric cards (Attacker, Defender, Network status, Nash equilibrium)
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("<div class='card'><div class='metric'>Attacker</div><div class='metric-label'>Current Strategy Mix</div></div>", unsafe_allow_html=True)
    st.metric(label="Low Attack", value=f"{attacker_mix[0]:.0%}")
    st.metric(label="High Attack", value=f"{attacker_mix[1]:.0%}")
with col2:
    st.markdown("<div class='card'><div class='metric'>Defender</div><div class='metric-label'>Current Strategy Mix</div></div>", unsafe_allow_html=True)
    st.metric(label="Low Defense", value=f"{defender_mix[0]:.0%}")
    st.metric(label="High Defense", value=f"{defender_mix[1]:.0%}")
with col3:
    st.markdown("<div class='card'><div class='metric'>Network</div><div class='metric-label'>Status Overview</div></div>", unsafe_allow_html=True)
    # Simple derived indicators (these are illustrative, not part of the model)
    traffic = 100 * (1 - (defender_mix[0] * 0.2 + defender_mix[1] * 0.5))
    st.metric(label="Traffic %", value=f"{traffic:.0f}%")
    st.metric(label="Damage", value=f"{damage:.1f}")
    st.metric(label="Under Attack", value="Yes" if traffic > 70 else "No")
with col4:
    st.markdown("<div class='card'><div class='metric'>Nash Equilibrium</div><div class='metric-label'>Result</div></div>", unsafe_allow_html=True)
    if pure_eq:
        eq_text = ", ".join([f"{a} + {d}" for a, d in pure_eq])
        st.success(eq_text)
    else:
        st.info("Mixed strategy only")
        st.write(
            f"Attacker: Low {attacker_mix[0]:.0%}, High {attacker_mix[1]:.0%}\n"
            f"Defender: Low {defender_mix[0]:.0%}, High {defender_mix[1]:.0%}"
        )

# ---------------------------------------------------------------------------
# Main navigation tabs
# ---------------------------------------------------------------------------
tab_game, tab_matrix, tab_theory, tab_sim = st.tabs(["🎮 Game", "📊 Payoff Matrix", "🧠 Game Theory", "🔄 Simulation"])

# ---------------------------------------------------------------------------
# Tab 1 – Visual Game Flow (Attacker → Network → Defender)
# ---------------------------------------------------------------------------
with tab_game:
    st.subheader("Current Interaction")
    with st.container():
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            st.markdown("""
<div class='card' style='background:#1e222a; text-align:center;'>
    <h4 style='color:#ff6b6b;'>Attacker</h4>
    <p>{'High Attack' if attacker_mix[1] > attacker_mix[0] else 'Low Attack'}</p>
</div>
""", unsafe_allow_html=True)
        with c2:
            st.markdown("""
<div class='card' style='background:#1e222a; text-align:center;'>
    <h4 style='color:#e0e0e0;'>Target Network</h4>
    <p>Traffic: {traffic:.0f}%</p>
    <p>Damage: {damage:.1f}</p>
    <p>Status: {'Under Attack' if traffic > 70 else 'Secure'}</p>
</div>
""", unsafe_allow_html=True)
        with c3:
            st.markdown("""
<div class='card' style='background:#1e222a; text-align:center;'>
    <h4 style='color:#4dd0e1;'>Defender</h4>
    <p>{'High Defense' if defender_mix[1] > defender_mix[0] else 'Low Defense'}</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab 2 – Payoff Matrix (combined view with highlights)
# ---------------------------------------------------------------------------
with tab_matrix:
    st.subheader("Payoff Matrix")
    # Build a combined DataFrame showing (Attacker, Defender) tuple per cell as strings
    combined = pd.DataFrame(index=attacker_df.index, columns=attacker_df.columns, dtype=str)
    for r in combined.index:
        for c in combined.columns:
            a_val = attacker_df.loc[r, c]
            d_val = defender_df.loc[r, c]
            combined.loc[r, c] = f"({a_val:.1f}, {d_val:.1f})"
    # Display the combined matrix
    st.table(combined)
    st.caption("Each cell shows (Attacker Payoff, Defender Payoff).")

# ---------------------------------------------------------------------------
# Tab 3 – Game Theory analysis (human‑readable best responses and equilibrium)
# ---------------------------------------------------------------------------
with tab_theory:
    st.subheader("Best Responses")
    # Attacker side
    for d_strategy, a_resps in attacker_br.items():
        st.markdown(f"**Against {d_strategy}:**")
        for a in a_resps:
            st.write(f"→ {a}")
    st.markdown("---")
    # Defender side
    for a_strategy, d_resps in defender_br.items():
        st.markdown(f"**Against {a_strategy}:**")
        for d in d_resps:
            st.write(f"→ {d}")

    st.subheader("Nash Equilibrium")
    if pure_eq:
        for a, d in pure_eq:
            st.success(f"🟢 {a} + {d}")
        st.caption("Both players have no incentive to deviate from this pair of pure strategies.")
    else:
        st.info("Mixed‑strategy equilibrium only.")
        st.write(
            f"Attacker: Low {attacker_mix[0]:.0%}, High {attacker_mix[1]:.0%}\n"
            f"Defender: Low {defender_mix[0]:.0%}, High {defender_mix[1]:.0%}"
        )
        st.caption("Probabilities make the opponent indifferent between their two options.")

# ---------------------------------------------------------------------------
# Tab 4 – Simulation (interactive with charts)
# ---------------------------------------------------------------------------
with tab_sim:
    st.subheader("Run Simulation")
    sim_rounds = st.slider("Number of rounds", min_value=10, max_value=500, value=50, step=10)
    if st.button("▶ Run Simulation"):
        sim_df = simulate_game(
            attacker_mix,
            defender_mix,
            attacker_df,
            defender_df,
            defence_effectiveness,
            rounds=sim_rounds,
            seed=42,
        )
        # Summary metrics
        col_a, col_d, col_s = st.columns(3)
        with col_a:
            st.metric("Avg Attacker Payoff", f"{sim_df['Attacker Payoff'].mean():.2f}")
        with col_d:
            st.metric("Avg Defender Payoff", f"{sim_df['Defender Payoff'].mean():.2f}")
        with col_s:
            st.metric("Attack Success Rate", f"{sim_df['Success'].mean():.0%}")

        # Table (first 10 rows)
        st.dataframe(sim_df.head(10))

        # Charts inside cards
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        # Cumulative success
        cum_success = sim_df["Success"].cumsum() / np.arange(1, len(sim_df) + 1)
        axs[0, 0].plot(cum_success, color="#ff6b6b")
        axs[0, 0].set_title("Cumulative Success Rate")
        # Attacker payoff over time
        axs[0, 1].plot(sim_df["Attacker Payoff"].values, color="#00ffab")
        axs[0, 1].set_title("Attacker Payoff")
        # Defender payoff over time
        axs[1, 0].plot(sim_df["Defender Payoff"].values, color="#4dd0e1")
        axs[1, 0].set_title("Defender Payoff")
        # Strategy frequencies
        att_counts = sim_df["Attacker"].value_counts().reindex(["Low Attack", "High Attack"]).fillna(0)
        def_counts = sim_df["Defender"].value_counts().reindex(["Low Defense", "High Defense"]).fillna(0)
        width = 0.35
        axs[1, 1].bar(np.arange(len(att_counts)), att_counts, width, label="Attacker", color="#ff6b6b")
        axs[1, 1].bar(np.arange(len(def_counts)) + width, def_counts, width, label="Defender", color="#4dd0e1")
        axs[1, 1].set_xticks(np.arange(len(att_counts)) + width / 2)
        axs[1, 1].set_xticklabels(att_counts.index, rotation=45)
        axs[1, 1].set_title("Strategy Frequencies")
        axs[1, 1].legend()
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

st.caption("All underlying calculations are transparent; see `game.py` for source code.")
