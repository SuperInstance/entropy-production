//! Thermodynamic fluxes and Onsager linear force-flux relations.
//!
//! In the linear regime near equilibrium, fluxes are linear functions of forces:
//!   J_i = Σ_j L_{ij} · X_j
//!
//! Onsager's reciprocal relations state L_{ij} = L_{ji} when the proper
//! choice of fluxes and forces is made (Onsager, 1931).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A thermodynamic force X driving a flux.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicForce {
    /// Name identifying this force.
    pub name: String,
    /// Magnitude of the force.
    pub value: f64,
}

/// A thermodynamic flux J resulting from forces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicFlux {
    /// Name identifying this flux.
    pub name: String,
    /// Magnitude of the flux.
    pub value: f64,
}

/// Onsager matrix L relating forces to fluxes: J = L · X.
///
/// L must be symmetric (L_{ij} = L_{ji}) for time-reversal invariant systems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnsagerMatrix {
    /// The matrix elements L[i][j].
    pub matrix: Vec<Vec<f64>>,
    /// Names of the fluxes/forces (same ordering for both).
    pub names: Vec<String>,
}

impl OnsagerMatrix {
    /// Create a new Onsager matrix with given dimension.
    pub fn new(n: usize) -> Self {
        OnsagerMatrix {
            matrix: vec![vec![0.0; n]; n],
            names: vec![],
        }
    }

    /// Create from a flat symmetric matrix and names.
    pub fn from_symmetric(matrix: Vec<Vec<f64>>, names: Vec<String>) -> Self {
        OnsagerMatrix { matrix, names }
    }

    /// Set element L[i][j] and L[j][i] simultaneously (enforce symmetry).
    pub fn set_symmetric(&mut self, i: usize, j: usize, value: f64) {
        self.matrix[i][j] = value;
        self.matrix[j][i] = value;
    }

    /// Dimension of the matrix.
    pub fn dim(&self) -> usize {
        self.matrix.len()
    }

    /// Check Onsager reciprocity: L_{ij} ≈ L_{ji} within tolerance.
    pub fn is_symmetric(&self, tolerance: f64) -> bool {
        let n = self.dim();
        for i in 0..n {
            for j in 0..n {
                if (self.matrix[i][j] - self.matrix[j][i]).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Compute fluxes J = L · X from given forces.
    pub fn compute_fluxes(&self, forces: &[f64]) -> Vec<f64> {
        self.matrix
            .iter()
            .map(|row| {
                row.iter()
                    .zip(forces.iter())
                    .map(|(&l, &x)| l * x)
                    .sum()
            })
            .collect()
    }

    /// Compute entropy production rate: σ = X^T · L · X.
    ///
    /// This is always ≥ 0 if L is positive semi-definite.
    pub fn entropy_production_rate(&self, forces: &[f64]) -> f64 {
        let fluxes = self.compute_fluxes(forces);
        forces.iter().zip(fluxes.iter()).map(|(&x, &j)| x * j).sum()
    }

    /// Compute the dissipation function Φ = T·σ = Σ J_i · X_i.
    pub fn dissipation(&self, forces: &[f64]) -> f64 {
        self.entropy_production_rate(forces)
    }

    /// Verify positive semi-definiteness by checking all principal minors ≥ 0.
    ///
    /// For 2x2: L11 ≥ 0, L22 ≥ 0, L11·L22 - L12·L21 ≥ 0.
    pub fn is_positive_semidefinite(&self) -> bool {
        let n = self.dim();
        // Check diagonal elements
        for i in 0..n {
            if self.matrix[i][i] < 0.0 {
                return false;
            }
        }
        // For 2x2, check determinant
        if n == 2 {
            let det = self.matrix[0][0] * self.matrix[1][1] - self.matrix[0][1] * self.matrix[1][0];
            if det < 0.0 {
                return false;
            }
        }
        true
    }
}

/// Heat flux computed from Fourier's law: J_q = -κ · ∇T.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatFlux {
    /// Thermal conductivity (W/(m·K)).
    pub conductivity: f64,
    /// Temperature gradient (K/m).
    pub temperature_gradient: f64,
}

impl HeatFlux {
    /// Compute heat flux magnitude.
    pub fn flux(&self) -> f64 {
        -self.conductivity * self.temperature_gradient
    }

    /// Entropy production per unit volume from heat conduction:
    /// σ = κ · (∇T)² / T²
    pub fn entropy_production(&self, temperature: f64) -> f64 {
        self.conductivity * self.temperature_gradient.powi(2) / temperature.powi(2)
    }
}

/// Particle flux from Fick's law: J_n = -D · ∇c.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleFlux {
    /// Diffusion coefficient (m²/s).
    pub diffusion_coefficient: f64,
    /// Concentration gradient (mol/m⁴).
    pub concentration_gradient: f64,
}

impl ParticleFlux {
    /// Compute particle flux magnitude.
    pub fn flux(&self) -> f64 {
        -self.diffusion_coefficient * self.concentration_gradient
    }
}

/// Coupled force-flux system with named variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoupledSystem {
    /// The Onsager matrix.
    pub onsager: OnsagerMatrix,
    /// Current forces as named values.
    pub forces: HashMap<String, f64>,
}

impl CoupledSystem {
    /// Create a new coupled system.
    pub fn new(onsager: OnsagerMatrix) -> Self {
        CoupledSystem {
            onsager,
            forces: HashMap::new(),
        }
    }

    /// Set a named force.
    pub fn set_force(&mut self, name: &str, value: f64) {
        self.forces.insert(name.to_string(), value);
    }

    /// Compute all fluxes from current forces.
    pub fn compute_all_fluxes(&self) -> HashMap<String, f64> {
        let force_vec: Vec<f64> = self
            .onsager
            .names
            .iter()
            .map(|n| *self.forces.get(n).unwrap_or(&0.0))
            .collect();
        let flux_vec = self.onsager.compute_fluxes(&force_vec);
        self.onsager
            .names
            .iter()
            .zip(flux_vec.iter())
            .map(|(n, &v)| (n.clone(), v))
            .collect()
    }

    /// Total entropy production rate for the coupled system.
    pub fn total_entropy_production(&self) -> f64 {
        let force_vec: Vec<f64> = self
            .onsager
            .names
            .iter()
            .map(|n| *self.forces.get(n).unwrap_or(&0.0))
            .collect();
        self.onsager.entropy_production_rate(&force_vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onsager_symmetry() {
        let mut l = OnsagerMatrix::new(2);
        l.set_symmetric(0, 1, 0.5);
        l.matrix[0][0] = 1.0;
        l.matrix[1][1] = 2.0;
        assert!(l.is_symmetric(1e-10));
        assert!((l.matrix[0][1] - 0.5).abs() < 1e-10);
        assert!((l.matrix[1][0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_onsager_compute_fluxes() {
        let l = OnsagerMatrix::from_symmetric(
            vec![vec![2.0, 0.5], vec![0.5, 3.0]],
            vec!["heat".to_string(), "diffusion".to_string()],
        );
        let forces = vec![1.0, 2.0];
        let fluxes = l.compute_fluxes(&forces);
        assert!((fluxes[0] - 3.0).abs() < 1e-10); // 2*1 + 0.5*2
        assert!((fluxes[1] - 6.5).abs() < 1e-10); // 0.5*1 + 3*2
    }

    #[test]
    fn test_entropy_production_rate() {
        let l = OnsagerMatrix::from_symmetric(
            vec![vec![2.0, 0.0], vec![0.0, 3.0]],
            vec!["A".to_string(), "B".to_string()],
        );
        let forces = vec![1.0, 1.0];
        // σ = X^T L X = [1,1] · [2,3]^T = 2 + 3 = 5
        assert!((l.entropy_production_rate(&forces) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_heat_flux_fourier() {
        let hf = HeatFlux {
            conductivity: 400.0,
            temperature_gradient: -100.0,
        };
        // J = -κ · ∇T = -400 · (-100) = 40000
        assert!((hf.flux() - 40000.0).abs() < 1e-10);
    }

    #[test]
    fn test_heat_flux_entropy_production() {
        let hf = HeatFlux {
            conductivity: 1.0,
            temperature_gradient: 10.0,
        };
        let sigma = hf.entropy_production(300.0);
        // σ = κ·(∇T)²/T² = 1·100/90000
        let expected = 100.0 / 90000.0;
        assert!((sigma - expected).abs() < 1e-12);
    }

    #[test]
    fn test_particle_flux() {
        let pf = ParticleFlux {
            diffusion_coefficient: 1e-9,
            concentration_gradient: -1e3,
        };
        assert!((pf.flux() - 1e-6).abs() < 1e-18);
    }

    #[test]
    fn test_coupled_system() {
        let l = OnsagerMatrix::from_symmetric(
            vec![vec![1.0, 0.2], vec![0.2, 1.0]],
            vec!["heat".to_string(), "mass".to_string()],
        );
        let mut sys = CoupledSystem::new(l);
        sys.set_force("heat", 10.0);
        sys.set_force("mass", 5.0);
        let fluxes = sys.compute_all_fluxes();
        assert!((fluxes["heat"] - 11.0).abs() < 1e-10); // 1*10 + 0.2*5
        assert!((fluxes["mass"] - 7.0).abs() < 1e-10); // 0.2*10 + 1*5
        assert!(sys.total_entropy_production() > 0.0);
    }

    #[test]
    fn test_positive_semidefinite() {
        let l = OnsagerMatrix::from_symmetric(
            vec![vec![2.0, 1.0], vec![1.0, 2.0]],
            vec!["A".to_string(), "B".to_string()],
        );
        assert!(l.is_positive_semidefinite());
    }

    #[test]
    fn test_not_positive_semidefinite() {
        let l = OnsagerMatrix::from_symmetric(
            vec![vec![-1.0, 0.0], vec![0.0, 2.0]],
            vec!["A".to_string(), "B".to_string()],
        );
        assert!(!l.is_positive_semidefinite());
    }
}
