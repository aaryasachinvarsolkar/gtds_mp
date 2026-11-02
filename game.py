import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

# ---------------------------------------------------------------------------
# Helper functions for the DDoS Attack‑Defense game
# ---------------------------------------------------------------------------

def build_payoff_matrices(
    attack_cost_low: float,
    attack_cost_high: float,
    defense_cost_low: float,
    defense_cost_high: float,
    damage: float,
    defence_effectiveness: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Construct attacker and defender payoff matrices.

    Strategies are ordered as:
        Attacker – [Low Attack, High Attack]
        Defender – [Low Defense, High Defense]

    The defender's high defense blocks a high‑attack with probability
    ``defence_effectiveness``; otherwise the attack succeeds.
    Low defense never blocks.

    Returns
    -------
    attacker_matrix, defender_matrix : pandas.DataFrames
        DataFrames indexed by attacker strategy and columns by defender strategy.
    """
    # Attacker payoff = damage (if attack succeeds) - attack cost
    # Defender payoff = -damage (if attack succeeds) - defense cost
    # Probability of success depends on strategies.
    prob_success = np.array([
        # Defender Low,   Defender High
        [1.0, 1.0],            # Attacker Low (always succeeds, low impact)
        [1.0, 1.0 - defence_effectiveness]  # Attacker High (blocked partly)
    ])

    # Costs vectors
    attack_costs = np.array([attack_cost_low, attack_cost_high])
    defense_costs = np.array([defense_cost_low, defense_cost_high])

    # Compute attacker payoff matrix
    attacker_matrix = np.zeros((2, 2))
    defender_matrix = np.zeros((2, 2))
    for i in range(2):  # attacker strategy
        for j in range(2):  # defender strategy
            success = prob_success[i, j]
            attacker_payoff = success * damage - attack_costs[i]
            defender_payoff = -success * damage - defense_costs[j]
            attacker_matrix[i, j] = attacker_payoff
            defender_matrix[i, j] = defender_payoff

    rows = ["Low Attack", "High Attack"]
    cols = ["Low Defense", "High Defense"]
    attacker_df = pd.DataFrame(attacker_matrix, index=rows, columns=cols)
    defender_df = pd.DataFrame(defender_matrix, index=rows, columns=cols)
    return attacker_df, defender_df


def best_responses(attacker_df: pd.DataFrame, defender_df: pd.DataFrame) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Calculate best‑response sets for both players.

    Returns
    -------
    attacker_br, defender_br : dict
        Mapping from opponent strategy to list of own best responses.
    """
    # Attacker best response to each defender column
    attacker_br = {}
    for col in attacker_df.columns:
        max_val = attacker_df[col].max()
        attacker_br[col] = attacker_df[col][attacker_df[col] == max_val].index.tolist()

    # Defender best response to each attacker row
    defender_br = {}
    for row in defender_df.index:
        max_val = defender_df.loc[row].max()
        defender_br[row] = defender_df.loc[row][defender_df.loc[row] == max_val].index.tolist()

    return attacker_br, defender_br


def pure_nash(attacker_br: Dict[str, List[str]], defender_br: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    """Identify pure‑strategy Nash equilibria.

    A profile (a, d) is a Nash equilibrium if:
        a ∈ attacker_br[d] and d ∈ defender_br[a]
    """
    equilibria = []
    for d in attacker_br:
        for a in attacker_br[d]:
            if d in defender_br[a]:
                equilibria.append((a, d))
    return equilibria


def mixed_nash(attacker_df: pd.DataFrame, defender_df: pd.DataFrame) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute mixed‑strategy Nash equilibrium analytically.

    The attacker mixes with probability ``p`` on Low Attack (1‑p on High).
    The defender mixes with probability ``q`` on Low Defense (1‑q on High).

    The equilibrium makes each player indifferent between their two strategies:
        * Attacker: Expected payoff of Low = Expected payoff of High
        * Defender: Expected payoff of Low = Expected payoff of High
    Solving the two linear equations yields (p, q).

    Returns
    -------
    (p_low, p_high), (q_low, q_high)
    """
    # Extract payoffs as numpy arrays
    A = attacker_df.values  # shape (2,2)
    D = defender_df.values

    # Indifference conditions
    # Attacker:   p*A[0,0] + (1-p)*A[1,0] = p*A[0,1] + (1-p)*A[1,1]
    # Defender:   q*D[0,0] + (1-q)*D[0,1] = q*D[1,0] + (1-q)*D[1,1]
    # Solve for p and q
    # Rearranged: p*(A00 - A01 - A10 + A11) = A11 - A01
    a_num = A[1, 1] - A[0, 1]
    a_den = (A[0, 0] - A[0, 1] - A[1, 0] + A[1, 1])
    p_low = a_num / a_den if a_den != 0 else np.nan

    # Defender similar
    d_num = D[1, 1] - D[1, 0]
    d_den = (D[0, 0] - D[0, 1] - D[1, 0] + D[1, 1])
    q_low = d_num / d_den if d_den != 0 else np.nan

    # Clip to [0,1]
    p_low = np.clip(p_low, 0, 1) if not np.isnan(p_low) else np.nan
    q_low = np.clip(q_low, 0, 1) if not np.isnan(q_low) else np.nan
    p_high = 1 - p_low if not np.isnan(p_low) else np.nan
    q_high = 1 - q_low if not np.isnan(q_low) else np.nan
    return (p_low, p_high), (q_low, q_high)


def simulate_game(
    attacker_mix: Tuple[float, float],
    defender_mix: Tuple[float, float],
    attacker_df: pd.DataFrame,
    defender_df: pd.DataFrame,
    defence_effectiveness: float,
    rounds: int = 50,
    seed: int | None = None,
) -> pd.DataFrame:
    """Run a repeated‑round simulation using the mixed strategies.

    Parameters
    ----------
    attacker_mix, defender_mix : tuple of floats
        Probabilities for (Low, High) strategies of attacker and defender.
    attacker_df, defender_df : DataFrames
        Payoff matrices (used for reporting payoffs).
    defence_effectiveness : float
        Probability that a high defense blocks a high attack.
    rounds : int, default 50
        Number of simulated rounds.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame with columns:
        Round, Attacker, Defender, Success, Attacker Payoff, Defender Payoff
    """
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(1, rounds + 1):
        a_choice = rng.choice(["Low Attack", "High Attack"], p=attacker_mix)
        d_choice = rng.choice(["Low Defense", "High Defense"], p=defender_mix)
        # Success probability according to the game model
        if a_choice == "High Attack" and d_choice == "High Defense":
            prob_success = 1.0 - defence_effectiveness
        else:
            prob_success = 1.0
        success = rng.random() < prob_success
        a_payoff = attacker_df.loc[a_choice, d_choice]
        d_payoff = defender_df.loc[a_choice, d_choice]
        rows.append({
            "Round": r,
            "Attacker": a_choice,
            "Defender": d_choice,
            "Success": success,
            "Attacker Payoff": a_payoff,
            "Defender Payoff": d_payoff,
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# End of module
# ---------------------------------------------------------------------------
