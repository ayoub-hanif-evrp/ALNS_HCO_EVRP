"""Feature dimension constants for the exact SMDP dynamic architecture."""
from __future__ import annotations

ARCHITECTURE_VERSION = "exact_smdp_dynamic_v1"

# Static node features (unchanged Schneider representation).
NODE_FEATURE_DIM = 9

# Acting EV dynamic vector (Section 5).
ACTING_EV_FEATURE_DIM = 9

# Per-EV communication token scalars before node embedding concat (Section 6).
EV_TOKEN_SCALAR_DIM = 7

# Global problem state (Section 9).
GLOBAL_STATE_FEATURE_DIM = 20

# Mask difficulty summary (Section 10).
MASK_SUMMARY_FEATURE_DIM = 10

# Fleet imbalance scalars (Section 6).
FLEET_BALANCE_FEATURE_DIM = 4

# Per-candidate dynamic features (Section 7).
# Dim 21 historical + wait_mu_norm + density_score for Hanif stations.
DYNAMIC_CANDIDATE_FEAT_DIM = 23

# Battery scalars for the charging head (Section 13).
BATTERY_STATE_FEATURE_DIM = 4
