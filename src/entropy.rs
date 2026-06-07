//! Entropy computations: Shannon entropy, joint entropy, conditional entropy,
//! mutual information, and relative entropy (KL divergence).
//!
//! All quantities computed from frequency counts over discrete distributions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A discrete probability distribution derived from frequency counts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Distribution {
    /// Probabilities for each outcome (always sums to 1.0).
    pub probabilities: Vec<f64>,
}

impl Distribution {
    /// Build a distribution from raw frequency counts.
    ///
    /// Counts are normalized so they sum to 1.0. Zero counts are preserved as
    /// zero probabilities but excluded from entropy summation.
    pub fn from_counts(counts: &[f64]) -> Self {
        let total: f64 = counts.iter().sum();
        if total == 0.0 {
            return Distribution {
                probabilities: vec![0.0; counts.len()],
            };
        }
        let probabilities: Vec<f64> = counts.iter().map(|c| c / total).collect();
        Distribution { probabilities }
    }

    /// Compute the Shannon entropy H = -Σ p_i · ln(p_i).
    ///
    /// Uses natural logarithm (nats). Returns 0.0 for degenerate distributions.
    pub fn shannon_entropy(&self) -> f64 {
        self.probabilities
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum()
    }

    /// Compute Shannon entropy using log-base-2 (bits).
    pub fn shannon_entropy_base2(&self) -> f64 {
        self.probabilities
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.log2())
            .sum()
    }

    /// Number of outcomes.
    pub fn len(&self) -> usize {
        self.probabilities.len()
    }

    /// Whether the distribution is empty.
    pub fn is_empty(&self) -> bool {
        self.probabilities.is_empty()
    }
}

/// Joint probability distribution over two discrete random variables X and Y.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointDistribution {
    /// Joint probability table: joint[i][j] = P(X=i, Y=j).
    pub joint: Vec<Vec<f64>>,
    /// Number of outcomes for X.
    pub n_x: usize,
    /// Number of outcomes for Y.
    pub n_y: usize,
}

impl JointDistribution {
    /// Build a joint distribution from a 2D frequency table.
    ///
    /// All entries are normalized by the total sum.
    pub fn from_counts(counts: &[Vec<f64>]) -> Self {
        let total: f64 = counts.iter().flat_map(|row| row.iter()).sum();
        let n_x = counts.len();
        let n_y = if n_x > 0 { counts[0].len() } else { 0 };
        if total == 0.0 {
            return JointDistribution {
                joint: vec![vec![0.0; n_y]; n_x],
                n_x,
                n_y,
            };
        }
        let joint: Vec<Vec<f64>> = counts
            .iter()
            .map(|row| row.iter().map(|c| c / total).collect())
            .collect();
        JointDistribution { joint, n_x, n_y }
    }

    /// Marginal distribution of X: P(X=i) = Σ_j P(X=i, Y=j).
    pub fn marginal_x(&self) -> Distribution {
        let probs: Vec<f64> = self.joint.iter().map(|row| row.iter().sum()).collect();
        Distribution {
            probabilities: probs,
        }
    }

    /// Marginal distribution of Y: P(Y=j) = Σ_i P(X=i, Y=j).
    pub fn marginal_y(&self) -> Distribution {
        let mut probs = vec![0.0; self.n_y];
        for row in &self.joint {
            for (j, &p) in row.iter().enumerate() {
                probs[j] += p;
            }
        }
        Distribution {
            probabilities: probs,
        }
    }

    /// Joint entropy H(X,Y) = -Σ_{i,j} P(x_i, y_j) · ln(P(x_i, y_j)).
    pub fn joint_entropy(&self) -> f64 {
        self.joint
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum()
    }

    /// Conditional entropy H(X|Y) = H(X,Y) - H(Y).
    pub fn conditional_entropy_x_given_y(&self) -> f64 {
        self.joint_entropy() - self.marginal_y().shannon_entropy()
    }

    /// Conditional entropy H(Y|X) = H(X,Y) - H(X).
    pub fn conditional_entropy_y_given_x(&self) -> f64 {
        self.joint_entropy() - self.marginal_x().shannon_entropy()
    }
}

/// Mutual information I(X;Y) = H(X) + H(Y) - H(X,Y).
///
/// Measures the amount of information one variable contains about the other.
pub fn mutual_information(joint: &JointDistribution) -> f64 {
    let hx = joint.marginal_x().shannon_entropy();
    let hy = joint.marginal_y().shannon_entropy();
    let hxy = joint.joint_entropy();
    hx + hy - hxy
}

/// Relative entropy (Kullback-Leibler divergence) D_KL(P || Q).
///
/// D_KL(P||Q) = Σ_i P(i) · ln(P(i) / Q(i)).
///
/// Returns `None` if any P(i) > 0 has Q(i) = 0 (divergence is infinite).
pub fn relative_entropy(p: &Distribution, q: &Distribution) -> Option<f64> {
    if p.len() != q.len() {
        return None;
    }
    let mut kl = 0.0;
    for (&pi, &qi) in p.probabilities.iter().zip(q.probabilities.iter()) {
        if pi > 0.0 {
            if qi <= 0.0 {
                return None; // KL divergence is infinite
            }
            kl += pi * (pi / qi).ln();
        }
    }
    Some(kl)
}

/// Compute Shannon entropy directly from a probability slice.
///
/// Convenience function: H = -Σ p_i · ln(p_i).
pub fn shannon_entropy(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

/// Build a joint distribution from paired observations.
///
/// Takes two slices of discrete observations and builds the joint frequency table.
pub fn joint_from_observations(
    x: &[usize],
    y: &[usize],
    n_x: usize,
    n_y: usize,
) -> JointDistribution {
    let mut counts = vec![vec![0.0; n_y]; n_x];
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        if xi < n_x && yi < n_y {
            counts[xi][yi] += 1.0;
        }
    }
    JointDistribution::from_counts(&counts)
}

/// Compute entropy from a frequency count map.
///
/// Keys are labels, values are occurrence counts.
pub fn entropy_from_map(counts: &HashMap<String, f64>) -> f64 {
    let total: f64 = counts.values().sum();
    if total == 0.0 {
        return 0.0;
    }
    counts
        .values()
        .filter(|&&c| c > 0.0)
        .map(|&c| {
            let p = c / total;
            -p * p.ln()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shannon_entropy_uniform() {
        let d = Distribution::from_counts(&[1.0, 1.0, 1.0, 1.0]);
        let h = d.shannon_entropy();
        assert!((h - (4.0f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_shannon_entropy_deterministic() {
        let d = Distribution::from_counts(&[1.0, 0.0, 0.0]);
        assert!(d.shannon_entropy().abs() < 1e-10);
    }

    #[test]
    fn test_shannon_entropy_coin() {
        let d = Distribution::from_counts(&[1.0, 1.0]);
        assert!((d.shannon_entropy() - (2.0f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_base2() {
        let d = Distribution::from_counts(&[1.0, 1.0]);
        assert!((d.shannon_entropy_base2() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_from_map() {
        let mut map = HashMap::new();
        map.insert("A".to_string(), 2.0);
        map.insert("B".to_string(), 2.0);
        let h = entropy_from_map(&map);
        assert!((h - (2.0f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_joint_entropy_independent() {
        // X and Y independent uniform on {0,1}
        let joint = JointDistribution::from_counts(&[vec![1.0, 1.0], vec![1.0, 1.0]]);
        let hxy = joint.joint_entropy();
        let hx = joint.marginal_x().shannon_entropy();
        let hy = joint.marginal_y().shannon_entropy();
        assert!((hxy - hx - hy).abs() < 1e-10);
    }

    #[test]
    fn test_mutual_information_correlated() {
        let joint = JointDistribution::from_counts(&[vec![10.0, 0.0], vec![0.0, 10.0]]);
        let mi = mutual_information(&joint);
        assert!(mi > 0.0);
        // Perfect correlation: MI = H(X) = H(Y)
        let hx = joint.marginal_x().shannon_entropy();
        assert!((mi - hx).abs() < 1e-10);
    }

    #[test]
    fn test_mutual_information_independent() {
        let joint = JointDistribution::from_counts(&[vec![1.0, 1.0], vec![1.0, 1.0]]);
        let mi = mutual_information(&joint);
        assert!(mi.abs() < 1e-10);
    }

    #[test]
    fn test_conditional_entropy() {
        let joint = JointDistribution::from_counts(&[vec![10.0, 0.0], vec![0.0, 10.0]]);
        // Perfect correlation: H(X|Y) = 0
        let hxy = joint.conditional_entropy_x_given_y();
        assert!(hxy.abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence_same() {
        let p = Distribution::from_counts(&[1.0, 1.0]);
        let q = Distribution::from_counts(&[1.0, 1.0]);
        assert!((relative_entropy(&p, &q).unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence_different() {
        let p = Distribution::from_counts(&[3.0, 1.0]);
        let q = Distribution::from_counts(&[1.0, 1.0]);
        let kl = relative_entropy(&p, &q).unwrap();
        assert!(kl > 0.0);
    }

    #[test]
    fn test_kl_divergence_infinite() {
        let p = Distribution::from_counts(&[1.0, 1.0]);
        let q = Distribution::from_counts(&[1.0, 0.0]);
        assert!(relative_entropy(&p, &q).is_none());
    }

    #[test]
    fn test_joint_from_observations() {
        let x = vec![0, 1, 0, 1];
        let y = vec![0, 1, 0, 1];
        let joint = joint_from_observations(&x, &y, 2, 2);
        let mi = mutual_information(&joint);
        assert!(mi > 0.1);
    }

    #[test]
    fn test_distribution_from_zero_counts() {
        let d = Distribution::from_counts(&[0.0, 0.0, 0.0]);
        assert_eq!(d.shannon_entropy(), 0.0);
    }

    #[test]
    fn test_kl_divergence_mismatched_lengths() {
        let p = Distribution::from_counts(&[1.0, 1.0]);
        let q = Distribution::from_counts(&[1.0, 1.0, 1.0]);
        assert!(relative_entropy(&p, &q).is_none());
    }
}
