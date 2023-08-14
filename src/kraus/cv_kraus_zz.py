import numpy as np
from forest.benchmarking.operator_tools import pauli_liouville2kraus

alpha = 1.36
cutoff = 20
error_channel = np.load(f"data/ptm/ptm_error_channel_rzz_alpha_{alpha}_cutoff_{cutoff}.npy")

# Error channel
R = error_channel

# Make sure R is trace preserving
r = np.zeros(16)
r[0] = 1
R[0,:] = r

# Convert PTM to Kraus
kraus_operators = pauli_liouville2kraus(R)

# Identity
id = sum(np.conj(k.T)@k for k in kraus_operators)
# Check that the kraus sum to identity
if np.isclose(id, np.eye(4), rtol=1e-4, atol=1e-4).all() != True:
    print(id)
    raise ValueError("Kraus operators must sum to identity")

# Save results
file = f"data/kraus/cv_kraus_rzz_alpha_{alpha}_cutoff_{cutoff}.npy"
np.save(file, kraus_operators)
