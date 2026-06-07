//! Thermodynamic network: graph of coupled processes.
//!
//! Nodes represent thermodynamic variables (temperatures, concentrations, etc.)
//! and edges represent fluxes between them. Each edge has an associated entropy
//! production rate.
//!
//! Following Schnakenberg (1976), the total entropy production of the network
//! is the sum of edge productions: σ_total = Σ_e J_e · X_e ≥ 0.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A node in the thermodynamic network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermoNode {
    /// Unique identifier.
    pub id: String,
    /// Physical quantity name (e.g., "temperature", "concentration").
    pub quantity_type: String,
    /// Current value of the variable.
    pub value: f64,
}

/// An edge representing a flux between two nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermoEdge {
    /// Unique identifier.
    pub id: String,
    /// Source node id.
    pub source: String,
    /// Target node id.
    pub target: String,
    /// Conductance (Onsager coefficient for this edge).
    pub conductance: f64,
    /// Current flux magnitude.
    pub flux: f64,
    /// Thermodynamic force (negative gradient).
    pub force: f64,
}

impl ThermoEdge {
    /// Compute entropy production rate for this edge: σ = J · X.
    pub fn entropy_production_rate(&self) -> f64 {
        self.flux * self.force
    }

    /// Update flux from force using linear relation: J = L · X.
    pub fn update_flux(&mut self) {
        self.flux = self.conductance * self.force;
    }
}

/// A thermodynamic network of coupled processes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicNetwork {
    /// Nodes in the network.
    pub nodes: HashMap<String, ThermoNode>,
    /// Edges in the network.
    pub edges: Vec<ThermoEdge>,
}

impl Default for ThermodynamicNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl ThermodynamicNetwork {
    /// Create an empty network.
    pub fn new() -> Self {
        ThermodynamicNetwork {
            nodes: HashMap::new(),
            edges: Vec::new(),
        }
    }

    /// Add a node to the network.
    pub fn add_node(&mut self, node: ThermoNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Add an edge to the network.
    pub fn add_edge(&mut self, edge: ThermoEdge) {
        self.edges.push(edge);
    }

    /// Compute forces for all edges from node values.
    ///
    /// Force X = -(value_target - value_source) / T (for heat)
    /// or simply X = value_source - value_target for generalized potential.
    pub fn compute_forces(&mut self) {
        for edge in &mut self.edges {
            let source_val = self.nodes.get(&edge.source).map(|n| n.value).unwrap_or(0.0);
            let target_val = self.nodes.get(&edge.target).map(|n| n.value).unwrap_or(0.0);
            edge.force = source_val - target_val;
        }
    }

    /// Update all fluxes from current forces.
    pub fn update_fluxes(&mut self) {
        for edge in &mut self.edges {
            edge.update_flux();
        }
    }

    /// Total entropy production rate: σ = Σ_e J_e · X_e.
    pub fn total_entropy_production(&self) -> f64 {
        self.edges.iter().map(|e| e.entropy_production_rate()).sum()
    }

    /// Entropy production rate at each edge.
    pub fn edge_productions(&self) -> Vec<(String, f64)> {
        self.edges
            .iter()
            .map(|e| (e.id.clone(), e.entropy_production_rate()))
            .collect()
    }

    /// Find the minimum dissipation routing for a given total flux.
    ///
    /// Uses iterative redistribution: edges with lower entropy production per
    /// unit flux get more flux allocated. Iterates until convergence.
    pub fn minimum_dissipation_routing(
        &self,
        total_flux: f64,
        tolerance: f64,
        max_iterations: usize,
    ) -> MinimumDissipationResult {
        let n_edges = self.edges.len();
        if n_edges == 0 {
            return MinimumDissipationResult {
                edge_fluxes: vec![],
                total_dissipation: 0.0,
                converged: true,
            };
        }

        // Start with equal distribution
        let mut fluxes = vec![total_flux / n_edges as f64; n_edges];

        let mut converged = false;
        for _ in 0..max_iterations {
            // Compute marginal dissipation for each edge: dΦ/dJ = 2·L·J
            let marginal_dissipation: Vec<f64> = self
                .edges
                .iter()
                .zip(fluxes.iter())
                .map(|(e, &j)| 2.0 * e.conductance * j)
                .collect();

            // Shift flux from high-marginal to low-marginal edges
            let avg_marginal: f64 = marginal_dissipation.iter().sum::<f64>() / n_edges as f64;
            if avg_marginal == 0.0 {
                converged = true;
                break;
            }

            let mut total_shift = 0.0;
            for (i, &marg) in marginal_dissipation.iter().enumerate() {
                let diff = marg - avg_marginal;
                let shift = 0.01 * diff / avg_marginal * fluxes[i];
                let new_flux = (fluxes[i] - shift).max(0.0);
                total_shift += (new_flux - fluxes[i]).abs();
                fluxes[i] = new_flux;
            }

            // Renormalize to maintain total flux
            let total: f64 = fluxes.iter().sum();
            if total > 0.0 {
                for f in &mut fluxes {
                    *f *= total_flux / total;
                }
            }

            if total_shift / total_flux < tolerance {
                converged = true;
                break;
            }
        }

        let total_dissipation: f64 = self
            .edges
            .iter()
            .zip(fluxes.iter())
            .map(|(e, &j)| e.conductance * j * j)
            .sum();

        MinimumDissipationResult {
            edge_fluxes: self
                .edges
                .iter()
                .zip(fluxes.iter())
                .map(|(e, &j)| (e.id.clone(), j))
                .collect(),
            total_dissipation,
            converged,
        }
    }

    /// Compute the network adjacency structure.
    pub fn adjacency(&self) -> HashMap<String, Vec<String>> {
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        for edge in &self.edges {
            adj.entry(edge.source.clone())
                .or_default()
                .push(edge.target.clone());
            adj.entry(edge.target.clone())
                .or_default()
                .push(edge.source.clone());
        }
        adj
    }

    /// Get all edges connected to a node.
    pub fn edges_for_node(&self, node_id: &str) -> Vec<&ThermoEdge> {
        self.edges
            .iter()
            .filter(|e| e.source == node_id || e.target == node_id)
            .collect()
    }
}

/// Result of minimum dissipation routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimumDissipationResult {
    /// Flux allocated to each edge.
    pub edge_fluxes: Vec<(String, f64)>,
    /// Total dissipation Φ = Σ L_e · J_e².
    pub total_dissipation: f64,
    /// Whether the iterative solver converged.
    pub converged: bool,
}

/// Build a simple linear chain network.
///
/// Nodes are numbered N0, N1, ..., N(n-1) with values from the slice.
/// Edges connect consecutive nodes with the given conductances.
pub fn linear_chain(values: &[f64], conductances: &[f64]) -> ThermodynamicNetwork {
    let mut net = ThermodynamicNetwork::new();

    for (i, &val) in values.iter().enumerate() {
        net.add_node(ThermoNode {
            id: format!("N{i}"),
            quantity_type: "potential".to_string(),
            value: val,
        });
    }

    for (i, &cond) in conductances.iter().enumerate() {
        if i + 1 < values.len() {
            net.add_edge(ThermoEdge {
                id: format!("E{i}"),
                source: format!("N{i}"),
                target: format!("N{}", i + 1),
                conductance: cond,
                flux: 0.0,
                force: 0.0,
            });
        }
    }

    net.compute_forces();
    net.update_fluxes();
    net
}

/// Build a star network: one central node connected to n peripheral nodes.
pub fn star_network(
    center_value: f64,
    center_id: &str,
    peripheral: &[(String, f64, f64)], // (id, value, conductance)
) -> ThermodynamicNetwork {
    let mut net = ThermodynamicNetwork::new();
    net.add_node(ThermoNode {
        id: center_id.to_string(),
        quantity_type: "potential".to_string(),
        value: center_value,
    });

    for (i, (pid, pval, cond)) in peripheral.iter().enumerate() {
        net.add_node(ThermoNode {
            id: pid.clone(),
            quantity_type: "potential".to_string(),
            value: *pval,
        });
        net.add_edge(ThermoEdge {
            id: format!("E{i}"),
            source: center_id.to_string(),
            target: pid.clone(),
            conductance: *cond,
            flux: 0.0,
            force: 0.0,
        });
    }

    net.compute_forces();
    net.update_fluxes();
    net
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_entropy_production() {
        let edge = ThermoEdge {
            id: "E0".to_string(),
            source: "A".to_string(),
            target: "B".to_string(),
            conductance: 2.0,
            flux: 3.0,
            force: 1.5,
        };
        assert!((edge.entropy_production_rate() - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_linear_chain_network() {
        let net = linear_chain(&[400.0, 350.0, 300.0], &[1.0, 1.5]);
        assert_eq!(net.nodes.len(), 3);
        assert_eq!(net.edges.len(), 2);
        // Force on E0 = 400 - 350 = 50
        assert!((net.edges[0].force - 50.0).abs() < 1e-10);
        // Flux on E0 = 1.0 * 50 = 50
        assert!((net.edges[0].flux - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_total_entropy_production() {
        let mut net = ThermodynamicNetwork::new();
        net.add_node(ThermoNode {
            id: "A".to_string(),
            quantity_type: "temp".to_string(),
            value: 400.0,
        });
        net.add_node(ThermoNode {
            id: "B".to_string(),
            quantity_type: "temp".to_string(),
            value: 300.0,
        });
        net.add_edge(ThermoEdge {
            id: "E0".to_string(),
            source: "A".to_string(),
            target: "B".to_string(),
            conductance: 1.0,
            flux: 0.0,
            force: 0.0,
        });
        net.compute_forces();
        net.update_fluxes();
        // J = 100, X = 100, σ = 10000
        assert!((net.total_entropy_production() - 10000.0).abs() < 1e-10);
    }

    #[test]
    fn test_edge_productions() {
        let net = linear_chain(&[100.0, 50.0, 0.0], &[1.0, 1.0]);
        let prods = net.edge_productions();
        assert_eq!(prods.len(), 2);
        // Both edges: force=50, flux=50, σ=2500
        for (_, sigma) in prods {
            assert!(sigma > 0.0);
        }
    }

    #[test]
    fn test_minimum_dissipation_two_paths() {
        let mut net = ThermodynamicNetwork::new();
        net.add_node(ThermoNode {
            id: "S".to_string(),
            quantity_type: "temp".to_string(),
            value: 100.0,
        });
        net.add_node(ThermoNode {
            id: "T".to_string(),
            quantity_type: "temp".to_string(),
            value: 0.0,
        });
        // Two parallel paths with different conductances
        net.add_edge(ThermoEdge {
            id: "fast".to_string(),
            source: "S".to_string(),
            target: "T".to_string(),
            conductance: 3.0,
            flux: 0.0,
            force: 0.0,
        });
        net.add_edge(ThermoEdge {
            id: "slow".to_string(),
            source: "S".to_string(),
            target: "T".to_string(),
            conductance: 1.0,
            flux: 0.0,
            force: 0.0,
        });

        let result = net.minimum_dissipation_routing(100.0, 1e-8, 500);
        // Minimum dissipation of L*J² with fixed total: J_e ∝ 1/L_e
        // Higher conductance path gets LESS flux (minimizes L·J²)
        let fast_flux: f64 = result
            .edge_fluxes
            .iter()
            .find(|(id, _)| id == "fast")
            .map(|(_, f)| *f)
            .unwrap();
        let slow_flux: f64 = result
            .edge_fluxes
            .iter()
            .find(|(id, _)| id == "slow")
            .map(|(_, f)| *f)
            .unwrap();
        // With L_fast=3, L_slow=1: optimal J_fast = 100*(1/3)/(1/3+1) = 25
        // J_slow = 100*1/(1/3+1) = 75
        assert!(
            slow_flux > fast_flux,
            "slow_flux={slow_flux} should exceed fast_flux={fast_flux}"
        );
    }

    #[test]
    fn test_star_network() {
        let net = star_network(
            100.0,
            "center",
            &[
                ("p1".to_string(), 80.0, 1.0),
                ("p2".to_string(), 60.0, 2.0),
                ("p3".to_string(), 40.0, 1.5),
            ],
        );
        assert_eq!(net.nodes.len(), 4);
        assert_eq!(net.edges.len(), 3);
        assert!(net.total_entropy_production() > 0.0);
    }

    #[test]
    fn test_adjacency() {
        let net = linear_chain(&[100.0, 50.0, 0.0], &[1.0, 1.0]);
        let adj = net.adjacency();
        assert!(adj.contains_key("N0"));
        assert!(adj.contains_key("N1"));
        assert!(adj.contains_key("N2"));
        assert!(adj["N1"].contains(&"N0".to_string()));
        assert!(adj["N1"].contains(&"N2".to_string()));
    }

    #[test]
    fn test_edges_for_node() {
        let net = linear_chain(&[100.0, 50.0, 0.0], &[1.0, 1.0]);
        let edges = net.edges_for_node("N1");
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_empty_network_production() {
        let net = ThermodynamicNetwork::new();
        assert_eq!(net.total_entropy_production(), 0.0);
    }
}
