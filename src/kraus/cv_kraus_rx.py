import numpy as np
from forest.benchmarking.operator_tools import pauli_liouville2kraus

alpha = 2
cutoff = 20
error_channel = np.load(f"data/ptm/ptm_error_channel_rx_alpha_{alpha}_cutoff_{cutoff}.npy")

# Angle of rotation
arg_list = np.linspace(0, np.pi, num=181, endpoint=False)

# Initialise list
kraus_list = []
for idx, arg in enumerate(arg_list):
    # Error channel
    R = error_channel[idx,:,:]
    # Make sure R is trace preserving
    R[0,:] = [1,0,0,0]

    # Convert PTM to Kraus
    kraus_operators = pauli_liouville2kraus(R)

    # Identity
    id = sum(np.conj(k.T)@k for k in kraus_operators)
    # Check that the kraus sum to identity
    if np.isclose(id, np.eye(2), rtol=1e-4, atol=1e-4).all() != True:
        print(id)
        raise ValueError("Kraus operators must sum to identity")
    
    # Append kraus to list
    kraus_list.append(kraus_operators)

kraus_list = np.asanyarray(kraus_list)
# Save results
file = f"data/kraus/cv_kraus_rx_alpha_{alpha}_cutoff_{cutoff}.npz"
np.savez(file, args=arg_list, kraus=kraus_list)