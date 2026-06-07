//! # entropy-production
//!
//! Non-equilibrium thermodynamics: entropy production and irreversible processes.
//!
//! This crate provides tools for computing entropy, entropy production rates,
//! thermodynamic fluxes, Onsager reciprocal relations, steady-state analysis,
//! chemical affinities, and thermodynamic network optimization.
//!
//! ## Modules
//!
//! - [`entropy`] — Shannon entropy, joint/conditional entropy, mutual information, KL divergence
//! - [`production`] — Entropy production in irreversible processes
//! - [`flux`] — Thermodynamic fluxes and Onsager linear force-flux relations
//! - [`steady_state`] — Prigogine's minimum entropy production principle
//! - [`affinity`] — Chemical affinity and reaction driving forces
//! - [`network`] — Thermodynamic network of coupled processes

pub mod affinity;
pub mod entropy;
pub mod flux;
pub mod network;
pub mod production;
pub mod steady_state;
