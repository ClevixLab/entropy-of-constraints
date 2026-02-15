"""
Example 2: Neural Network Training with Adaptive Learning Rates
================================================================

MATHEMATICAL MODEL:
------------------
Parameter dynamics:
    dw_i/dt = -θ_i ∂L/∂w_i
    
Adaptive learning rate dynamics:
    dθ_i/dt = -α θ_i (∂L/∂w_i)² + β ξ_i(t)

where:
    w ∈ ℝᵈ: network parameters
    θ ∈ ℝᵈ₊: component-wise learning rates
    L(w): loss function
    α: learning rate decay (reinforcement)
    β: exploration noise (novelty)

PHYSICAL INTERPRETATION:
-----------------------
- w: Where we are in parameter space
- θ: How fast we move in each direction
- High gradient → θ decreases → slow down in that direction
- Result: Network "learns" to ignore certain directions

DUAL COLLAPSE:
-------------
1. Parameters converge to minimum (H_w decreases)
2. Learning rates concentrate (H_θ decreases)
3. Effective dimensionality << d (low-dim manifold)

Author: Truong Xuan Khanh
"""

import numpy as np
from scipy.integrate import solve_ivp
import sys
sys.path.append('../core')
from eoc_system import EoCSystem

class NeuralNetworkEoC(EoCSystem):
    """Neural network with adaptive learning rates."""
    
    def __init__(
        self,
        d: int = 20,
        alpha: float = 0.01,
        beta: float = 0.001,
        loss_type: str = 'quadratic'
    ):
        super().__init__(n=d, m=d, alpha=alpha, beta=beta,
                        name="Neural Network with Adaptive Learning Rates")
        
        self.d = d
        self.loss_type = loss_type
        
        # Target for loss (optimal parameters)
        self.w_star = np.random.randn(d) * 0.1
        
        # Hessian approximation (for quadratic loss)
        if loss_type == 'quadratic':
            H = np.random.randn(d, d)
            self.H = (H + H.T) / 2 + np.eye(d) * 0.1  # Make positive definite
        
        # System parameters
        self.M0 = 1.0
        self.Mtheta = 0.5
        self.kappa = 1.0
        self.CN = 1.0
    
    def loss(self, w: np.ndarray) -> float:
        """Loss function L(w)."""
        if self.loss_type == 'quadratic':
            diff = w - self.w_star
            return 0.5 * diff @ self.H @ diff
        else:
            return 0.5 * np.sum((w - self.w_star)**2)
    
    def gradient(self, w: np.ndarray) -> np.ndarray:
        """Gradient ∇L(w)."""
        if self.loss_type == 'quadratic':
            return self.H @ (w - self.w_star)
        else:
            return w - self.w_star
    
    def F(self, w: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Parameter dynamics: dw/dt = -θ ⊙ ∇L."""
        grad = self.gradient(w)
        return -theta * grad
    
    def G(self, w: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Learning rate dynamics: dθ/dt = -α θ (∇L)² + β ξ."""
        grad = self.gradient(w)
        grad_squared = grad**2
        
        # Deterministic: ξ = 0
        noise = np.zeros_like(theta)
        
        return -self.alpha * theta * grad_squared + self.beta * noise
    
    def verify_assumptions(self, n_samples: int = 100) -> dict:
        """Verify EoC assumptions for neural network."""
        print("Verifying assumptions for Neural Network...")
        
        results = {}
        
        # Sample phase space
        w_samples = np.random.randn(n_samples, self.d) * 0.5
        theta_samples = np.random.uniform(0.01, 1.0, (n_samples, self.d))
        
        # Check k-volume contraction
        eigenvalues_list = []
        for i in range(min(50, n_samples)):
            w, theta = w_samples[i], theta_samples[i]
            
            # Jacobian of (F, G)
            grad = self.gradient(w)
            
            # Diagonal Jacobian (simplified)
            J_F = -np.diag(theta) @ self.H if hasattr(self, 'H') else -np.diag(theta)
            J_G = -self.alpha * np.diag(grad**2)
            
            # Combined Jacobian (block diagonal structure)
            eigs_F = np.linalg.eigvals(J_F)
            eigs_G = np.linalg.eigvals(J_G)
            
            eigenvalues_list.append(np.concatenate([eigs_F, eigs_G]))
        
        all_eigs = np.array(eigenvalues_list)
        mean_eigs = np.mean(all_eigs.real, axis=0)
        sorted_eigs = np.sort(mean_eigs)[::-1]
        
        # Find k where sum of top k eigenvalues < 0
        cumsum = np.cumsum(sorted_eigs)
        k_collapse = np.where(cumsum < 0)[0]
        
        if len(k_collapse) > 0:
            k = k_collapse[0] + 1
        else:
            k = 2 * self.d
        
        results['k_volume_contraction'] = {
            'k': k,
            'dim_bound': k,
            'satisfies': k < 2 * self.d
        }
        
        alpha_c = self.compute_critical_threshold()
        results['critical_threshold'] = {
            'alpha_c': alpha_c,
            'in_collapse': self.alpha > alpha_c
        }
        
        print(f"  k-volume: k = {k} < 2d = {2*self.d}")
        print(f"  α_c = {alpha_c:.4f}, current α = {self.alpha:.4f}")
        
        return results

if __name__ == "__main__":
    print("Neural Network EoC Model")
    system = NeuralNetworkEoC(d=10, alpha=0.02, beta=0.001)
    print(system)
    
    results = system.verify_assumptions()
    
    # Quick simulation
    w0 = np.random.randn(10) * 0.5
    theta0 = np.ones(10) * 0.1
    
    traj = system.simulate(w0, theta0, T=50, dt=0.5)
    print(f"\nSimulation: {traj['success']}")
    print(f"Final loss: {system.loss(traj['x'][:,-1]):.6f}")
