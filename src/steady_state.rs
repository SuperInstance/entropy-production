//! Steady-state analysis: Prigogine's minimum entropy production principle.
//!
//! In the linear non-equilibrium regime, a system evolves toward a steady state
//! that minimizes the rate of entropy production (Prigogine, 1947).
//!
//! Key results:
//! - At steady state, entropy production is minimized subject to constraints.
//! - Perturbations from steady state cause the system to relax back.
//! - This holds only in the linear regime (near equilibrium).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A linear non-equilibrium thermodynamic system.
///
/// Described by an Onsager matrix and a set of constrained and free forces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearNonEquilibriumSystem {
    /// Onsager matrix L (n×n).
    pub onsager_matrix: Vec<Vec<f64>>,
    /// Names of the n forces.
    pub force_names: Vec<String>,
    /// Constrained forces (fixed by boundary conditions): name → value.
    pub constrained_forces: HashMap<String, f64>,
    /// Convergence tolerance for iterative solver.
    pub tolerance: f64,
    /// Maximum iterations.
    pub max_iterations: usize,
}

impl LinearNonEquilibriumSystem {
    /// Create a new system with the given Onsager matrix.
    pub fn new(onsager_matrix: Vec<Vec<f64>>, force_names: Vec<String>) -> Self {
        LinearNonEquilibriumSystem {
            onsager_matrix,
            force_names,
            constrained_forces: HashMap::new(),
            tolerance: 1e-10,
            max_iterations: 1000,
        }
    }

    /// Constrain a force to a fixed value.
    pub fn constrain_force(&mut self, name: &str, value: f64) {
        self.constrained_forces.insert(name.to_string(), value);
    }

    /// Get indices of free (unconstrained) forces.
    fn free_indices(&self) -> Vec<usize> {
        self.force_names
            .iter()
            .enumerate()
            .filter(|(_, name)| !self.constrained_forces.contains_key(*name))
            .map(|(i, _)| i)
            .collect()
    }

    /// Compute entropy production rate σ = X^T · L · X.
    pub fn entropy_production(&self, forces: &[f64]) -> f64 {
        let n = forces.len();
        let mut sigma = 0.0;
        for i in 0..n {
            for j in 0..n {
                sigma += forces[i] * self.onsager_matrix[i][j] * forces[j];
            }
        }
        sigma
    }

    /// Compute fluxes J = L · X.
    pub fn compute_fluxes(&self, forces: &[f64]) -> Vec<f64> {
        self.onsager_matrix
            .iter()
            .map(|row| {
                row.iter()
                    .zip(forces.iter())
                    .map(|(&l, &x)| l * x)
                    .sum()
            })
            .collect()
    }

    /// Find the steady state by minimizing entropy production.
    ///
    /// Uses iterative gradient descent on the free forces while keeping
    /// constrained forces fixed. At the minimum, ∂σ/∂X_free = 0, which
    /// means the corresponding fluxes vanish (minimum dissipation).
    ///
    /// Returns the steady-state forces and entropy production rate.
    pub fn find_steady_state(&self) -> SteadyStateResult {
        let n = self.force_names.len();
        let free_idx = self.free_indices();

        // Initialize forces
        let mut forces = vec![0.0; n];
        for (name, &val) in &self.constrained_forces {
            if let Some(idx) = self.force_names.iter().position(|n| n == name) {
                forces[idx] = val;
            }
        }
        // Initialize free forces to small random-ish values
        for &i in &free_idx {
            forces[i] = 0.01;
        }

        let learning_rate = 0.01;
        let mut prev_sigma = self.entropy_production(&forces);

        for _ in 0..self.max_iterations {
            // Compute gradient ∂σ/∂X_i = 2 * Σ_j L_{ij} * X_j
            for &i in &free_idx {
                let grad_i: f64 = self.onsager_matrix[i]
                    .iter()
                    .zip(forces.iter())
                    .map(|(&l, &x)| l * x)
                    .sum();
                forces[i] -= learning_rate * 2.0 * grad_i;
            }

            let sigma = self.entropy_production(&forces);
            if (sigma - prev_sigma).abs() < self.tolerance {
                break;
            }
            prev_sigma = sigma;
        }

        let sigma = self.entropy_production(&forces);
        let fluxes = self.compute_fluxes(&forces);

        SteadyStateResult {
            steady_forces: self
                .force_names
                .iter()
                .zip(forces.iter())
                .map(|(n, &v)| (n.clone(), v))
                .collect(),
            steady_fluxes: self
                .force_names
                .iter()
                .zip(fluxes.iter())
                .map(|(n, &v)| (n.clone(), v))
                .collect(),
            entropy_production_rate: sigma,
        }
    }

    /// Simulate perturbation from steady state and relaxation back.
    ///
    /// Returns a time series of (forces, entropy_production) showing the
    /// system relaxing back to the minimum entropy production state.
    pub fn perturbation_relaxation(
        &self,
        perturbation: &HashMap<String, f64>,
        dt: f64,
        n_steps: usize,
    ) -> Vec<RelaxationStep> {
        let steady = self.find_steady_state();
        let n = self.force_names.len();
        let free_idx = self.free_indices();

        // Start from perturbed state
        let mut forces = vec![0.0; n];
        for (name, &val) in &steady.steady_forces {
            if let Some(idx) = self.force_names.iter().position(|n| n == name) {
                forces[idx] = val;
            }
        }
        for (name, &delta) in perturbation {
            if let Some(idx) = self.force_names.iter().position(|n| n == name) {
                forces[idx] += delta;
            }
        }

        let mut trajectory = Vec::with_capacity(n_steps);

        for step in 0..n_steps {
            let sigma = self.entropy_production(&forces);
            let fluxes = self.compute_fluxes(&forces);

            trajectory.push(RelaxationStep {
                time: step as f64 * dt,
                forces: forces.clone(),
                entropy_production: sigma,
            });

            // Relaxation: dX/dt = -γ · J_free (free forces evolve toward zero flux)
            for &i in &free_idx {
                let relaxation_rate = 0.5;
                forces[i] -= relaxation_rate * fluxes[i] * dt;
            }
        }

        trajectory
    }
}

/// Result of finding a steady state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteadyStateResult {
    /// Steady-state force values.
    pub steady_forces: HashMap<String, f64>,
    /// Steady-state flux values.
    pub steady_fluxes: HashMap<String, f64>,
    /// Entropy production rate at steady state.
    pub entropy_production_rate: f64,
}

/// A single step in a relaxation trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelaxationStep {
    /// Time of this step.
    pub time: f64,
    /// Force values at this time.
    pub forces: Vec<f64>,
    /// Entropy production rate at this time.
    pub entropy_production: f64,
}

/// Verify that entropy production is minimized at steady state.
///
/// Compares σ at steady state with σ at nearby points to confirm it's a minimum.
pub fn verify_minimum(system: &LinearNonEquilibriumSystem, epsilon: f64) -> bool {
    let steady = system.find_steady_state();
    let sigma_steady = steady.entropy_production_rate;

    let n = system.force_names.len();
    let mut forces = vec![0.0; n];
    for (name, &val) in &steady.steady_forces {
        if let Some(idx) = system.force_names.iter().position(|nm| nm == name) {
            forces[idx] = val;
        }
    }

    // Check that perturbing any free force increases σ
    let free_idx: Vec<usize> = system
        .force_names
        .iter()
        .enumerate()
        .filter(|(_, name)| !system.constrained_forces.contains_key(*name))
        .map(|(i, _)| i)
        .collect();

    for &i in &free_idx {
        forces[i] += epsilon;
        let sigma_plus = system.entropy_production(&forces);
        forces[i] -= 2.0 * epsilon;
        let sigma_minus = system.entropy_production(&forces);
        forces[i] += epsilon; // restore

        if sigma_plus < sigma_steady - 1e-10 || sigma_minus < sigma_steady - 1e-10 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_steady_state_simple() {
        let mut sys = LinearNonEquilibriumSystem::new(
            vec![vec![2.0, 0.5], vec![0.5, 1.0]],
            vec!["X1".to_string(), "X2".to_string()],
        );
        sys.constrain_force("X1", 1.0);
        sys.tolerance = 1e-8;
        sys.max_iterations = 2000;

        let result = sys.find_steady_state();
        // At minimum: ∂σ/∂X2 = 0 → 0.5*X1 + 1.0*X2 = 0 → X2 = -0.5
        assert!((result.steady_forces["X2"] - (-0.5)).abs() < 0.1);
        assert!(result.entropy_production_rate >= 0.0);
    }

    #[test]
    fn test_entropy_production_minimum() {
        let mut sys = LinearNonEquilibriumSystem::new(
            vec![vec![3.0, 1.0], vec![1.0, 2.0]],
            vec!["A".to_string(), "B".to_string()],
        );
        sys.constrain_force("A", 2.0);
        assert!(verify_minimum(&sys, 0.01));
    }

    #[test]
    fn test_relaxation_decreases_entropy_production() {
        let mut sys = LinearNonEquilibriumSystem::new(
            vec![vec![2.0, 0.5], vec![0.5, 1.0]],
            vec!["X1".to_string(), "X2".to_string()],
        );
        sys.constrain_force("X1", 1.0);
        let mut perturbation = HashMap::new();
        perturbation.insert("X2".to_string(), 2.0);

        let trajectory = sys.perturbation_relaxation(&perturbation, 0.1, 50);
        // Entropy production should decrease over time
        assert!(
            trajectory.last().unwrap().entropy_production
                <= trajectory.first().unwrap().entropy_production + 1e-6
        );
    }

    #[test]
    fn test_compute_fluxes() {
        let sys = LinearNonEquilibriumSystem::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec!["A".to_string(), "B".to_string()],
        );
        let fluxes = sys.compute_fluxes(&[3.0, 4.0]);
        assert!((fluxes[0] - 3.0).abs() < 1e-10);
        assert!((fluxes[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_production_quadratic() {
        let sys = LinearNonEquilibriumSystem::new(
            vec![vec![2.0, 0.0], vec![0.0, 3.0]],
            vec!["A".to_string(), "B".to_string()],
        );
        // σ = 2*1² + 3*2² = 2 + 12 = 14
        let sigma = sys.entropy_production(&[1.0, 2.0]);
        assert!((sigma - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_3x3_steady_state() {
        let mut sys = LinearNonEquilibriumSystem::new(
            vec![
                vec![2.0, 0.5, 0.0],
                vec![0.5, 3.0, 1.0],
                vec![0.0, 1.0, 1.5],
            ],
            vec!["X1".to_string(), "X2".to_string(), "X3".to_string()],
        );
        sys.constrain_force("X1", 1.0);
        sys.max_iterations = 3000;

        let result = sys.find_steady_state();
        assert!(result.entropy_production_rate >= 0.0);
    }

    #[test]
    fn test_relaxation_steps_count() {
        let mut sys = LinearNonEquilibriumSystem::new(
            vec![vec![1.0, 0.2], vec![0.2, 1.0]],
            vec!["A".to_string(), "B".to_string()],
        );
        sys.constrain_force("A", 1.0);
        let mut pert = HashMap::new();
        pert.insert("B".to_string(), 0.5);

        let traj = sys.perturbation_relaxation(&pert, 0.1, 20);
        assert_eq!(traj.len(), 20);
    }
}
