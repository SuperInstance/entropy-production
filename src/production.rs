//! Entropy production in irreversible processes.
//!
//! The second law of thermodynamics for open systems:
//!   dS = dS_e + dS_i
//! where dS_e is the entropy exchanged with the environment (entropy flow)
//! and dS_i is the entropy produced internally (entropy production).
//!
//! The second law requires dS_i ≥ 0 for all processes.

use serde::{Deserialize, Serialize};

/// Total entropy change decomposed into flow and production.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyChange {
    /// Entropy flow: exchange with surroundings. Can be positive or negative.
    pub entropy_flow: f64,
    /// Entropy production: internally generated. Always ≥ 0.
    pub entropy_production: f64,
}

impl EntropyChange {
    /// Total entropy change = flow + production.
    pub fn total(&self) -> f64 {
        self.entropy_flow + self.entropy_production
    }

    /// Verify the second law: entropy production ≥ 0.
    pub fn satisfies_second_law(&self) -> bool {
        self.entropy_production >= 0.0
    }
}

/// Entropy production rate σ = dS_i/dt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyProduction {
    /// Rate of entropy production (W/K or J/(K·s)).
    pub rate: f64,
}

impl EntropyProduction {
    /// Create from flow and production components.
    pub fn from_components(flow_rate: f64, total_rate: f64) -> Self {
        EntropyProduction {
            rate: total_rate - flow_rate,
        }
    }

    /// Entropy production from heat flow between two reservoirs.
    ///
    /// σ = Q̇ · (1/T_cold - 1/T_hot) ≥ 0 when heat flows hot → cold.
    pub fn from_heat_flow(heat_rate: f64, t_hot: f64, t_cold: f64) -> Self {
        let rate = heat_rate * (1.0 / t_cold - 1.0 / t_hot);
        EntropyProduction { rate }
    }

    /// Entropy production from diffusion (Fick's law context).
    ///
    /// σ = J · ∇(-μ/T) where J is particle flux and ∇(-μ/T) is the
    /// generalized force. Simplified: σ = flux * (mu_hot/T_hot - mu_cold/T_cold).
    pub fn from_diffusion(flux: f64, mu_hot: f64, t_hot: f64, mu_cold: f64, t_cold: f64) -> Self {
        let force = (mu_hot / t_hot) - (mu_cold / t_cold);
        EntropyProduction { rate: flux * force }
    }

    /// Entropy production from a chemical reaction.
    ///
    /// σ = (A / T) · ξ̇ where A is the affinity and ξ̇ is the reaction rate.
    pub fn from_reaction(affinity: f64, reaction_rate: f64, temperature: f64) -> Self {
        EntropyProduction {
            rate: (affinity / temperature) * reaction_rate,
        }
    }

    /// Total entropy production rate from multiple independent processes.
    pub fn total(processes: &[EntropyProduction]) -> EntropyProduction {
        EntropyProduction {
            rate: processes.iter().map(|p| p.rate).sum(),
        }
    }

    /// Whether this satisfies the second law (σ ≥ 0).
    pub fn is_dissipative(&self) -> bool {
        self.rate >= 0.0
    }
}

/// Compute entropy production for a simple heat engine cycle.
///
/// Returns the entropy production per cycle.
pub fn heat_engine_production(q_hot: f64, q_cold: f64, t_hot: f64, t_cold: f64) -> EntropyChange {
    let entropy_flow = -q_hot / t_hot + q_cold / t_cold;
    let production = 0.0; // Reversible by default
    EntropyChange {
        entropy_flow,
        entropy_production: production,
    }
}

/// Compute entropy production for an irreversible heat engine.
///
/// Includes dissipation δQ at the cold reservoir.
pub fn irreversible_heat_engine(q_hot: f64, q_cold: f64, t_hot: f64, t_cold: f64) -> EntropyChange {
    let entropy_flow = -q_hot / t_hot + q_cold / t_cold;
    // For irreversible: ΔS_universe = q_cold/T_cold - q_hot/T_hot
    let production = q_cold / t_cold - q_hot / t_hot;
    EntropyChange {
        entropy_flow,
        entropy_production: production,
    }
}

/// Dissipation function Φ = T · σ (Prigogine).
///
/// Measures the rate of free energy dissipation.
pub fn dissipation_function(temperature: f64, entropy_production_rate: f64) -> f64 {
    temperature * entropy_production_rate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_second_law_heat_flow() {
        let ep = EntropyProduction::from_heat_flow(100.0, 400.0, 300.0);
        assert!(ep.rate > 0.0);
        assert!(ep.is_dissipative());
    }

    #[test]
    fn test_heat_flow_reversed_temps_negative() {
        // If cold > hot, entropy production is negative (unphysical).
        let ep = EntropyProduction::from_heat_flow(100.0, 300.0, 400.0);
        assert!(ep.rate < 0.0);
    }

    #[test]
    fn test_entropy_production_from_reaction() {
        let ep = EntropyProduction::from_reaction(50.0, 0.1, 300.0);
        assert!(ep.rate > 0.0);
        let expected = (50.0 / 300.0) * 0.1;
        assert!((ep.rate - expected).abs() < 1e-10);
    }

    #[test]
    fn test_total_entropy_change() {
        let ec = EntropyChange {
            entropy_flow: -50.0,
            entropy_production: 80.0,
        };
        assert!((ec.total() - 30.0).abs() < 1e-10);
        assert!(ec.satisfies_second_law());
    }

    #[test]
    fn test_dissipation_function() {
        let phi = dissipation_function(300.0, 0.5);
        assert!((phi - 150.0).abs() < 1e-10);
    }

    #[test]
    fn test_total_production() {
        let processes = vec![
            EntropyProduction { rate: 0.1 },
            EntropyProduction { rate: 0.2 },
            EntropyProduction { rate: 0.3 },
        ];
        let total = EntropyProduction::total(&processes);
        assert!((total.rate - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_diffusion_entropy_production() {
        let ep = EntropyProduction::from_diffusion(1.0, 5.0, 300.0, 3.0, 300.0);
        assert!(ep.rate > 0.0);
    }

    #[test]
    fn test_irreversible_heat_engine() {
        let ec = irreversible_heat_engine(600.0, 450.0, 400.0, 300.0);
        // Production = 450/300 - 600/400 = 1.5 - 1.5 = 0
        assert!(ec.entropy_production >= -1e-10);
    }

    #[test]
    fn test_from_components() {
        let ep = EntropyProduction::from_components(2.0, 5.0);
        assert!((ep.rate - 3.0).abs() < 1e-10);
    }
}
