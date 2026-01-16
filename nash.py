"""Nash equilibrium calculation using R-NAD (Replicator Neural Annealing Dynamics)."""
import numpy as np
from typing import Tuple, List


def rnad_replicator_step(x: np.ndarray, y: np.ndarray, M: np.ndarray,
                         pi_reg_row: np.ndarray, pi_reg_col: np.ndarray,
                         eta: float = 0.2, dt: float = 0.02, eps: float = 1e-12
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single step of R-NAD replicator dynamics with entropy regularization.

    Args:
        x: Row player strategy (probability distribution)
        y: Column player strategy (probability distribution)
        M: Payoff matrix (row player maximizes, column player minimizes)
        pi_reg_row: Reference distribution for row player regularization
        pi_reg_col: Reference distribution for column player regularization
        eta: Entropy regularization strength
        dt: Time step for integration
        eps: Small constant for numerical stability

    Returns:
        Updated (x, y) strategy pair
    """
    # Normalize inputs
    x = np.clip(x, eps, None)
    x = x / x.sum()
    y = np.clip(y, eps, None)
    y = y / y.sum()
    pi_reg_row = np.clip(pi_reg_row, eps, None)
    pi_reg_row = pi_reg_row / pi_reg_row.sum()
    pi_reg_col = np.clip(pi_reg_col, eps, None)
    pi_reg_col = pi_reg_col / pi_reg_col.sum()

    # Payoffs (row maximizes, column minimizes)
    q_row = M @ y
    q_col = -M.T @ x

    # Regularized fitness with entropy regularization
    f_row = q_row - eta * (np.log(x + eps) - np.log(pi_reg_row + eps))
    f_col = q_col - eta * (np.log(y + eps) - np.log(pi_reg_col + eps))

    # Replicator dynamics equation
    u_row = f_row - x @ f_row
    u_col = f_col - y @ f_col

    # Multiplicative Euler integration step
    x = x * np.exp(dt * u_row)
    x = x / x.sum()
    y = y * np.exp(dt * u_col)
    y = y / y.sum()

    return x, y


def compute_nash_equilibrium(payoff_matrix: np.ndarray,
                             num_restarts: int = 1,
                             inner_steps: int = 1000,
                             max_outer_iters: int = 10000,
                             eta: float = 0.2,
                             dt: float = 0.1,
                             tol: float = 1e-7
                             ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute Nash equilibrium of a zero-sum game using R-NAD with annealing.

    Uses multiple random restarts and selects the solution with maximum entropy.

    Args:
        payoff_matrix: Zero-sum payoff matrix (row player's payoffs)
        num_restarts: Number of random restarts
        inner_steps: Number of replicator steps before updating magnet
        max_outer_iters: Maximum outer iterations per restart
        eta: Entropy regularization parameter
        dt: Time step size
        tol: Convergence tolerance for magnet updates

    Returns:
        (row_strategy, col_strategy, game_value)
    """
    n = payoff_matrix.shape[0]
    m = payoff_matrix.shape[1]

    best_x, best_y = None, None
    best_entropy = -np.inf

    for restart in range(num_restarts):
        # Random initialization
        x = np.random.dirichlet(np.ones(n))
        y = np.random.dirichlet(np.ones(m))

        # Random magnet initialization
        pi_reg_row = np.random.dirichlet(np.ones(n))
        pi_reg_col = np.random.dirichlet(np.ones(m))

        # Outer loop: update magnet until convergence
        for outer in range(max_outer_iters):
            pi_old_row = pi_reg_row.copy()
            pi_old_col = pi_reg_col.copy()

            # Inner loop: run replicator dynamics
            for inner in range(inner_steps):
                x, y = rnad_replicator_step(x, y, payoff_matrix, pi_reg_row, pi_reg_col,
                                             eta=eta, dt=dt)

            # Update magnet to current strategy
            pi_reg_row = x.copy()
            pi_reg_col = y.copy()

            # Check convergence
            if (np.max(np.abs(pi_reg_row - pi_old_row)) < tol and
                np.max(np.abs(pi_reg_col - pi_old_col)) < tol):
                break

        # Calculate entropy of solution
        eps = 1e-12
        entropy = -np.sum(x * np.log(x + eps)) - np.sum(y * np.log(y + eps))

        # Keep solution with maximum entropy
        if entropy > best_entropy:
            best_entropy = entropy
            best_x = x.copy()
            best_y = y.copy()

    # Calculate game value
    game_value = best_x @ payoff_matrix @ best_y

    return best_x, best_y, game_value


def format_nash_strategy(strategy: np.ndarray, deck_names: List[str],
                         threshold: float = 0.01) -> str:
    """Format Nash strategy as human-readable string."""
    parts = []
    for i, prob in enumerate(strategy):
        if prob >= threshold:
            parts.append(f"{deck_names[i]}: {prob*100:.1f}%")
    return ", ".join(parts)
