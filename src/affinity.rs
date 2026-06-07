//! Chemical affinity: driving force for chemical reactions.
//!
//! The affinity A = -Σ ν_i · μ_i where ν_i are stoichiometric coefficients
//! and μ_i are chemical potentials. At equilibrium, A = 0.
//!
//! The reaction rate ξ̇ is proportional to the affinity (near equilibrium):
//!   ξ̇ = L · A
//! where L is the phenomenological coefficient.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A chemical species with its stoichiometric coefficient and chemical potential.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Species {
    /// Name of the species.
    pub name: String,
    /// Stoichiometric coefficient (negative for reactants, positive for products).
    pub stoichiometry: f64,
    /// Chemical potential μ (J/mol).
    pub chemical_potential: f64,
}

/// A chemical reaction with associated species.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChemicalReaction {
    /// Name of the reaction.
    pub name: String,
    /// Species involved in this reaction.
    pub species: Vec<Species>,
    /// Phenomenological coefficient L (relates affinity to rate).
    pub rate_coefficient: f64,
    /// Temperature at which the reaction occurs (K).
    pub temperature: f64,
}

impl ChemicalReaction {
    /// Compute the chemical affinity: A = -Σ ν_i · μ_i.
    ///
    /// Positive affinity means the reaction proceeds forward.
    pub fn affinity(&self) -> f64 {
        -self
            .species
            .iter()
            .map(|s| s.stoichiometry * s.chemical_potential)
            .sum::<f64>()
    }

    /// Compute the reaction rate: ξ̇ = L · A.
    pub fn reaction_rate(&self) -> f64 {
        self.rate_coefficient * self.affinity()
    }

    /// Entropy production from this reaction: σ = (A/T) · ξ̇.
    pub fn entropy_production(&self) -> f64 {
        let a = self.affinity();
        let xi_dot = self.reaction_rate();
        (a / self.temperature) * xi_dot
    }

    /// Check if the reaction is at equilibrium (A ≈ 0).
    pub fn is_at_equilibrium(&self, tolerance: f64) -> bool {
        self.affinity().abs() < tolerance
    }

    /// Create a reaction from stoichiometry and potential maps.
    pub fn from_maps(
        name: &str,
        stoichiometry: &HashMap<String, f64>,
        potentials: &HashMap<String, f64>,
        rate_coefficient: f64,
        temperature: f64,
    ) -> Self {
        let species: Vec<Species> = stoichiometry
            .iter()
            .map(|(name, &nu)| Species {
                name: name.clone(),
                stoichiometry: nu,
                chemical_potential: *potentials.get(name).unwrap_or(&0.0),
            })
            .collect();
        ChemicalReaction {
            name: name.to_string(),
            species,
            rate_coefficient,
            temperature,
        }
    }
}

/// System of coupled chemical reactions sharing intermediates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoupledReactionSystem {
    /// The reactions in the system.
    pub reactions: Vec<ChemicalReaction>,
}

impl CoupledReactionSystem {
    /// Create a new coupled reaction system.
    pub fn new(reactions: Vec<ChemicalReaction>) -> Self {
        CoupledReactionSystem { reactions }
    }

    /// Compute affinities for all reactions.
    pub fn all_affinities(&self) -> Vec<f64> {
        self.reactions.iter().map(|r| r.affinity()).collect()
    }

    /// Compute reaction rates for all reactions.
    pub fn all_rates(&self) -> Vec<f64> {
        self.reactions.iter().map(|r| r.reaction_rate()).collect()
    }

    /// Total entropy production from all reactions.
    pub fn total_entropy_production(&self) -> f64 {
        self.reactions.iter().map(|r| r.entropy_production()).sum()
    }

    /// Find shared intermediate species between reactions.
    pub fn shared_intermediates(&self) -> HashMap<String, Vec<usize>> {
        let mut intermediates: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, reaction) in self.reactions.iter().enumerate() {
            for species in &reaction.species {
                intermediates
                    .entry(species.name.clone())
                    .or_default()
                    .push(i);
            }
        }
        // Only keep species that appear in more than one reaction
        intermediates.retain(|_, indices| indices.len() > 1);
        intermediates
    }

    /// Simulate the reaction system over time using iterative Euler method.
    ///
    /// Chemical potentials evolve based on reaction rates.
    /// Returns trajectory of (time, potentials, affinities, entropy_production).
    pub fn simulate(
        &self,
        initial_potentials: &HashMap<String, f64>,
        susceptibilities: &HashMap<String, f64>,
        dt: f64,
        n_steps: usize,
    ) -> Vec<ReactionTrajectoryStep> {
        let mut potentials = initial_potentials.clone();
        let mut trajectory = Vec::with_capacity(n_steps);

        for step in 0..n_steps {
            // Build reactions with current potentials
            let mut current_reactions = self.reactions.clone();
            for reaction in &mut current_reactions {
                for species in &mut reaction.species {
                    if let Some(&mu) = potentials.get(&species.name) {
                        species.chemical_potential = mu;
                    }
                }
            }
            let current_system = CoupledReactionSystem::new(current_reactions);
            let affinities = current_system.all_affinities();
            let rates = current_system.all_rates();
            let sigma = current_system.total_entropy_production();

            trajectory.push(ReactionTrajectoryStep {
                time: step as f64 * dt,
                potentials: potentials.clone(),
                affinities: affinities.clone(),
                entropy_production: sigma,
            });

            // Update potentials: dμ_i/dt = -χ_i · Σ_ν ν_ij · ξ̇_j
            for (species_name, &chi) in susceptibilities {
                let mut dmu = 0.0;
                for (j, reaction) in current_system.reactions.iter().enumerate() {
                    for sp in &reaction.species {
                        if sp.name == *species_name {
                            dmu += sp.stoichiometry * rates[j];
                        }
                    }
                }
                let entry = potentials.entry(species_name.clone()).or_insert(0.0);
                *entry -= chi * dmu * dt;
            }
        }

        trajectory
    }
}

/// A step in a reaction trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactionTrajectoryStep {
    /// Time.
    pub time: f64,
    /// Chemical potentials at this time.
    pub potentials: HashMap<String, f64>,
    /// Affinities for each reaction.
    pub affinities: Vec<f64>,
    /// Total entropy production rate.
    pub entropy_production: f64,
}

/// Compute affinity from standard Gibbs energies.
///
/// A = -ΔG = -Σ ν_i · μ_i° - RT · ln(Q)
/// where Q is the reaction quotient.
pub fn affinity_from_standard_gibbs(
    stoichiometry: &HashMap<String, f64>,
    standard_potentials: &HashMap<String, f64>,
    concentrations: &HashMap<String, f64>,
    temperature: f64,
) -> f64 {
    let r = 8.314; // J/(mol·K)
    let mut a = 0.0;
    for (species, &nu) in stoichiometry {
        let mu0 = *standard_potentials.get(species).unwrap_or(&0.0);
        let c = *concentrations.get(species).unwrap_or(&1.0);
        a -= nu * (mu0 + r * temperature * c.ln());
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affinity_simple() {
        let reaction = ChemicalReaction {
            name: "A → B".to_string(),
            species: vec![
                Species {
                    name: "A".to_string(),
                    stoichiometry: -1.0,
                    chemical_potential: 1000.0,
                },
                Species {
                    name: "B".to_string(),
                    stoichiometry: 1.0,
                    chemical_potential: 500.0,
                },
            ],
            rate_coefficient: 0.01,
            temperature: 300.0,
        };
        // A = -(-1*1000 + 1*500) = 500
        assert!((reaction.affinity() - 500.0).abs() < 1e-10);
    }

    #[test]
    fn test_equilibrium_affinity_zero() {
        let reaction = ChemicalReaction {
            name: "equilibrium".to_string(),
            species: vec![
                Species {
                    name: "A".to_string(),
                    stoichiometry: -1.0,
                    chemical_potential: 500.0,
                },
                Species {
                    name: "B".to_string(),
                    stoichiometry: 1.0,
                    chemical_potential: 500.0,
                },
            ],
            rate_coefficient: 0.01,
            temperature: 300.0,
        };
        assert!(reaction.is_at_equilibrium(1e-10));
    }

    #[test]
    fn test_reaction_rate() {
        let reaction = ChemicalReaction {
            name: "test".to_string(),
            species: vec![
                Species {
                    name: "A".to_string(),
                    stoichiometry: -1.0,
                    chemical_potential: 1000.0,
                },
                Species {
                    name: "B".to_string(),
                    stoichiometry: 1.0,
                    chemical_potential: 200.0,
                },
            ],
            rate_coefficient: 0.05,
            temperature: 300.0,
        };
        // A = 800, rate = 0.05 * 800 = 40
        assert!((reaction.reaction_rate() - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_production_positive() {
        let reaction = ChemicalReaction {
            name: "exothermic".to_string(),
            species: vec![
                Species {
                    name: "A".to_string(),
                    stoichiometry: -1.0,
                    chemical_potential: 1000.0,
                },
                Species {
                    name: "B".to_string(),
                    stoichiometry: 1.0,
                    chemical_potential: 100.0,
                },
            ],
            rate_coefficient: 0.01,
            temperature: 300.0,
        };
        assert!(reaction.entropy_production() > 0.0);
    }

    #[test]
    fn test_from_maps() {
        let mut stoich = HashMap::new();
        stoich.insert("H2".to_string(), -2.0);
        stoich.insert("O2".to_string(), -1.0);
        stoich.insert("H2O".to_string(), 2.0);

        let mut pots = HashMap::new();
        pots.insert("H2".to_string(), -100.0);
        pots.insert("O2".to_string(), -200.0);
        pots.insert("H2O".to_string(), -300.0);

        let reaction = ChemicalReaction::from_maps("water synthesis", &stoich, &pots, 0.1, 298.0);
        // A = -(−2·(−100) + −1·(−200) + 2·(−300)) = -(200 + 200 − 600) = 200
        assert!((reaction.affinity() - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_coupled_system_shared_intermediates() {
        let r1 = ChemicalReaction {
            name: "R1: A → B".to_string(),
            species: vec![
                Species {
                    name: "A".to_string(),
                    stoichiometry: -1.0,
                    chemical_potential: 1000.0,
                },
                Species {
                    name: "B".to_string(),
                    stoichiometry: 1.0,
                    chemical_potential: 800.0,
                },
            ],
            rate_coefficient: 0.01,
            temperature: 300.0,
        };
        let r2 = ChemicalReaction {
            name: "R2: B → C".to_string(),
            species: vec![
                Species {
                    name: "B".to_string(),
                    stoichiometry: -1.0,
                    chemical_potential: 800.0,
                },
                Species {
                    name: "C".to_string(),
                    stoichiometry: 1.0,
                    chemical_potential: 600.0,
                },
            ],
            rate_coefficient: 0.02,
            temperature: 300.0,
        };
        let system = CoupledReactionSystem::new(vec![r1, r2]);
        let shared = system.shared_intermediates();
        assert!(shared.contains_key("B"));
        assert!(system.total_entropy_production() > 0.0);
    }

    #[test]
    fn test_simulation_trajectory() {
        let r1 = ChemicalReaction {
            name: "A→B".to_string(),
            species: vec![
                Species {
                    name: "A".to_string(),
                    stoichiometry: -1.0,
                    chemical_potential: 1000.0,
                },
                Species {
                    name: "B".to_string(),
                    stoichiometry: 1.0,
                    chemical_potential: 500.0,
                },
            ],
            rate_coefficient: 0.001,
            temperature: 300.0,
        };
        let system = CoupledReactionSystem::new(vec![r1]);
        let mut init_pots = HashMap::new();
        init_pots.insert("A".to_string(), 1000.0);
        init_pots.insert("B".to_string(), 500.0);
        let mut suscept = HashMap::new();
        suscept.insert("A".to_string(), 0.1);
        suscept.insert("B".to_string(), 0.1);

        let traj = system.simulate(&init_pots, &suscept, 0.1, 10);
        assert_eq!(traj.len(), 10);
        // Entropy production should decrease as system approaches equilibrium
        assert!(traj.last().unwrap().entropy_production >= 0.0);
    }

    #[test]
    fn test_affinity_from_standard_gibbs() {
        let mut stoich = HashMap::new();
        stoich.insert("A".to_string(), -1.0);
        stoich.insert("B".to_string(), 1.0);

        let mut std_pots = HashMap::new();
        std_pots.insert("A".to_string(), 0.0);
        std_pots.insert("B".to_string(), 0.0);

        let mut conc = HashMap::new();
        conc.insert("A".to_string(), 2.0);
        conc.insert("B".to_string(), 1.0);

        let a = affinity_from_standard_gibbs(&stoich, &std_pots, &conc, 300.0);
        // With equal standard potentials and [A]=2, [B]=1:
        // A = -(-1·RT·ln2 + 1·RT·ln1) = RT·ln2 > 0
        assert!(a > 0.0);
    }
}
