"""
Core EoC System Implementation
================================

This module provides the base class for all Entropy of Constraints systems.

Mathematical Framework:
----------------------
Coupled dynamics on M = X × Θ:
    dx/dt = F(x, θ)    [state evolution]
    dθ/dt = G(x, θ)    [constraint evolution]

where:
    x ∈ X ⊂ ℝⁿ     state variables
    θ ∈ Θ ⊂ ℝᵐ     constraint parameters
    α ≥ 0           reinforcement strength
    β ≥ 0           novelty strength

Key Decomposition:
    F = F₀ + α R    [baseline + reinforcement]
    G = G₀ - β N    [baseline - novelty]

Critical Thresholds:
    αc(β) = (M₀ + Mθ + βCN)/κ    [collapse threshold]
    βc = γ/(LCN)                  [invariance threshold]

Author: Truong Xuan Khanh
Date: February 2026
License: MIT
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.distance import pdist
from typing import Callable, Tuple, Optional, Dict, List
import warnings

class EoCSystem:
    """
    Base class for Entropy of Constraints dynamical systems.
    
    This class implements the mathematical framework for analyzing
    coupled state-constraint dynamics with entropy measures at both levels.
    
    Attributes
    ----------
    n : int
        State space dimension
    m : int  
        Constraint space dimension
    alpha : float
        Reinforcement strength parameter (≥ 0)
    beta : float
        Novelty strength parameter (≥ 0)
    
    Methods
    -------
    F(x, theta) : array
        State dynamics vector field
    G(x, theta) : array  
        Constraint dynamics vector field
    simulate(x0, theta0, T) : dict
        Integrate coupled ODE system
    compute_entropy(trajectory) : dict
        Calculate Shannon differential entropy
    estimate_dimension(trajectory) : float
        Estimate fractal dimension via box-counting
    verify_assumptions() : dict
        Check if system satisfies EoC assumptions
    compute_critical_threshold() : float
        Calculate αc(β) analytically or numerically
    """
    
    def __init__(
        self,
        n: int,
        m: int,
        alpha: float = 0.5,
        beta: float = 0.1,
        name: str = "Generic EoC System"
    ):
        """
        Initialize EoC system.
        
        Parameters
        ----------
        n : int
            State dimension (e.g., 3 for replicator with 3 strategies)
        m : int
            Constraint dimension (e.g., 1 for single selection parameter)
        alpha : float, optional
            Reinforcement strength (default: 0.5)
        beta : float, optional
            Novelty strength (default: 0.1)
        name : str, optional
            System identifier
        """
        self.n = n
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.name = name
        
        # Parameters for analysis (to be set by subclasses)
        self.M0 = None  # Bound on ∇·F₀
        self.Mtheta = None  # Bound on ∇·G₀
        self.kappa = None  # Reinforcement contraction rate
        self.CN = None  # Novelty bound
        
        # Validation flags
        self._assumptions_verified = False
        self._critical_threshold_computed = False
        
    def F(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        State dynamics: dx/dt = F(x, θ)
        
        MUST be overridden by subclass.
        
        Parameters
        ----------
        x : np.ndarray, shape (n,)
            Current state
        theta : np.ndarray, shape (m,)
            Current constraint parameters
            
        Returns
        -------
        dxdt : np.ndarray, shape (n,)
            State velocity
            
        Mathematical Form:
        -----------------
        F(x,θ) = F₀(x,θ) + α R(x,θ)
        
        where:
            F₀: baseline dynamics
            R: reinforcement (∇·R ≤ -κ)
        """
        raise NotImplementedError("Subclass must implement F(x, theta)")
    
    def G(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Constraint dynamics: dθ/dt = G(x, θ)
        
        MUST be overridden by subclass.
        
        Parameters
        ----------
        x : np.ndarray, shape (n,)
            Current state
        theta : np.ndarray, shape (m,)
            Current constraint parameters
            
        Returns
        -------
        dthetadt : np.ndarray, shape (m,)
            Constraint velocity
            
        Mathematical Form:
        -----------------
        G(x,θ) = G₀(x,θ) - β N(x,θ)
        
        where:
            G₀: baseline constraint evolution
            N: novelty injection (||∇·N|| ≤ CN)
        """
        raise NotImplementedError("Subclass must implement G(x, theta)")
    
    def coupled_dynamics(self, t: float, z: np.ndarray) -> np.ndarray:
        """
        Full coupled system: d/dt [x; θ] = [F; G]
        
        Parameters
        ----------
        t : float
            Time (for ODE solver interface)
        z : np.ndarray, shape (n+m,)
            Joint state [x; θ]
            
        Returns
        -------
        dzdt : np.ndarray, shape (n+m,)
            Joint velocity [dx/dt; dθ/dt]
        """
        x = z[:self.n]
        theta = z[self.n:]
        
        dxdt = self.F(x, theta)
        dthetadt = self.G(x, theta)
        
        return np.concatenate([dxdt, dthetadt])
    
    def simulate(
        self,
        x0: np.ndarray,
        theta0: np.ndarray,
        T: float = 100.0,
        dt: float = 0.1,
        method: str = 'RK45',
        rtol: float = 1e-6,
        atol: float = 1e-9
    ) -> Dict:
        """
        Simulate coupled EoC system.
        
        Parameters
        ----------
        x0 : np.ndarray, shape (n,)
            Initial state
        theta0 : np.ndarray, shape (m,)
            Initial constraint
        T : float, optional
            Simulation time horizon
        dt : float, optional
            Output time step
        method : str, optional
            ODE solver method (default: RK45)
        rtol, atol : float, optional
            Relative and absolute tolerances
            
        Returns
        -------
        result : dict
            Dictionary containing:
                - 't': np.ndarray, time points
                - 'x': np.ndarray, shape (n, len(t)), state trajectory  
                - 'theta': np.ndarray, shape (m, len(t)), constraint trajectory
                - 'H_X': np.ndarray, state entropy over time
                - 'H_C': np.ndarray, constraint entropy over time
                - 'success': bool, whether integration succeeded
                
        Mathematical Notes:
        ------------------
        Uses scipy.integrate.solve_ivp with adaptive stepping to ensure
        accuracy while handling potential stiffness in the coupled system.
        
        The entropy estimates H_X(t), H_C(t) are computed via kernel density
        estimation (KDE) on ensemble of trajectories. For single trajectory,
        use ensemble_simulate() instead.
        """
        # Initial condition
        z0 = np.concatenate([x0, theta0])
        
        # Time span
        t_span = (0, T)
        t_eval = np.arange(0, T, dt)
        
        # Integrate
        sol = solve_ivp(
            fun=self.coupled_dynamics,
            t_span=t_span,
            y0=z0,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            vectorized=False
        )
        
        if not sol.success:
            warnings.warn(f"Integration failed: {sol.message}")
        
        # Extract components
        x_traj = sol.y[:self.n, :]
        theta_traj = sol.y[self.n:, :]
        
        # Package results
        result = {
            't': sol.t,
            'x': x_traj,
            'theta': theta_traj,
            'success': sol.success,
            'message': sol.message if hasattr(sol, 'message') else ''
        }
        
        # Compute entropies (placeholder - requires ensemble)
        # For single trajectory, entropy is undefined
        # User should call ensemble_simulate() for entropy estimation
        result['H_X'] = None
        result['H_C'] = None
        
        return result
    
    def ensemble_simulate(
        self,
        n_trajectories: int = 1000,
        x0_distribution: str = 'uniform',
        theta0_distribution: str = 'uniform',
        **kwargs
    ) -> Dict:
        """
        Simulate ensemble of trajectories for entropy estimation.
        
        Parameters
        ----------
        n_trajectories : int
            Number of independent trajectories
        x0_distribution : str
            Initial state distribution ('uniform', 'gaussian')
        theta0_distribution : str  
            Initial constraint distribution
        **kwargs : dict
            Passed to simulate()
            
        Returns
        -------
        results : dict
            Ensemble results including entropy estimates
            
        Mathematical Background:
        -----------------------
        Differential entropy requires density p_t(x), which we estimate via:
        
        H_X(t) = -∫ p_t(x) log p_t(x) dx
        
        We use Kernel Density Estimation (KDE) with Gaussian kernels:
        
        p̂_t(x) = (1/n) Σᵢ K_h(x - xᵢ(t))
        
        where K_h is Gaussian kernel with bandwidth h chosen by Scott's rule.
        
        IMPORTANT: This is valid only during transient phase while
        p_t remains absolutely continuous. On attractor, use KS entropy instead.
        """
        # Sample initial conditions
        x0_samples = self._sample_initial_states(n_trajectories, x0_distribution)
        theta0_samples = self._sample_initial_constraints(n_trajectories, theta0_distribution)
        
        # Simulate all trajectories
        trajectories = []
        for i in range(n_trajectories):
            result = self.simulate(x0_samples[i], theta0_samples[i], **kwargs)
            if result['success']:
                trajectories.append(result)
        
        print(f"Successfully integrated {len(trajectories)}/{n_trajectories} trajectories")
        
        # Estimate entropies via KDE
        t = trajectories[0]['t']
        H_X_t = np.zeros(len(t))
        H_C_t = np.zeros(len(t))
        
        for i, time in enumerate(t):
            # Collect states and constraints at this time
            x_samples = np.array([traj['x'][:, i] for traj in trajectories])
            theta_samples = np.array([traj['theta'][:, i] for traj in trajectories])
            
            # Estimate entropy via KDE
            H_X_t[i] = self._estimate_entropy_kde(x_samples)
            H_C_t[i] = self._estimate_entropy_kde(theta_samples)
        
        return {
            't': t,
            'trajectories': trajectories,
            'H_X': H_X_t,
            'H_C': H_C_t,
            'n_success': len(trajectories)
        }
    
    def _sample_initial_states(self, n: int, distribution: str) -> np.ndarray:
        """Sample initial states from specified distribution."""
        if distribution == 'uniform':
            return np.random.uniform(-1, 1, (n, self.n))
        elif distribution == 'gaussian':
            return np.random.randn(n, self.n)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    
    def _sample_initial_constraints(self, n: int, distribution: str) -> np.ndarray:
        """Sample initial constraints from specified distribution."""
        if distribution == 'uniform':
            return np.random.uniform(0.01, 1, (n, self.m))
        elif distribution == 'gaussian':
            return np.abs(np.random.randn(n, self.m)) + 0.1
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    
    def _estimate_entropy_kde(self, samples: np.ndarray, bandwidth: str = 'scott') -> float:
        """
        Estimate Shannon differential entropy via KDE.
        
        H = -∫ p(x) log p(x) dx ≈ -E[log p̂(X)]
        
        Uses Gaussian kernel with Scott's bandwidth rule.
        """
        from scipy.stats import gaussian_kde
        
        if len(samples) < 10:
            return np.nan  # Too few samples
        
        try:
            kde = gaussian_kde(samples.T, bw_method=bandwidth)
            log_density = kde.logpdf(samples.T)
            entropy = -np.mean(log_density)
            return entropy
        except:
            return np.nan
    
    def compute_critical_threshold(self) -> float:
        """
        Compute critical reinforcement threshold αc(β).
        
        Returns
        -------
        alpha_c : float
            Critical threshold
            
        Mathematical Formula:
        --------------------
        αc(β) = (M₀ + Mθ + βCN) / κ
        
        where:
            M₀: bound on |∇·F₀|
            Mθ: bound on |∇·G₀|  
            κ: reinforcement contraction rate (∇·R ≤ -κ)
            CN: novelty bound (||∇·N|| ≤ CN)
            
        Interpretation:
        --------------
        When α > αc(β): reinforcement dominates → dual collapse
        When α < αc(β): novelty/baseline dominate → adaptive regime
        
        The threshold increases linearly with β:
            dαc/dβ = CN/κ
        meaning stronger novelty delays collapse.
        """
        if not self._assumptions_verified:
            warnings.warn("Assumptions not verified. Call verify_assumptions() first.")
        
        if self.M0 is None or self.Mtheta is None or self.kappa is None or self.CN is None:
            raise ValueError("System parameters not set. Cannot compute threshold.")
        
        alpha_c = (self.M0 + self.Mtheta + self.beta * self.CN) / self.kappa
        self._critical_threshold_computed = True
        
        return alpha_c
    
    def verify_assumptions(self, n_samples: int = 1000) -> Dict:
        """
        Numerically verify EoC assumptions.
        
        Checks:
        1. Dissipativity: ∃ Lyapunov function W with ∇W·(F,G) ≤ -c₀W + c₁
        2. Reinforcement contraction: ∇·R ≤ -κ on absorbing set
        3. Bounded novelty: ||∇·N|| ≤ CN
        
        Returns
        -------
        verification : dict
            Results of assumption checks
        """
        print(f"Verifying assumptions for {self.name}...")
        
        # To be implemented by subclass with specific verifications
        raise NotImplementedError("Subclass must implement verify_assumptions()")
    
    def __repr__(self) -> str:
        return (f"{self.name}\n"
                f"  State dim: {self.n}\n"
                f"  Constraint dim: {self.m}\n"
                f"  α (reinforcement): {self.alpha}\n"
                f"  β (novelty): {self.beta}")
