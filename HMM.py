# import numpy as np
# from hmmlearn import hmm
# import random



# # Define MFCC features for a few phonemes (e.g., "aa," "iy," "sh")
# aa_features = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
# iy_features = np.array([[4.0, 3.0, 2.0], [3.0, 2.0, 1.0], [2.0, 1.0, 0.0]])
# sh_features = np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])

# # Create HMM models for "aa," "iy," and "sh"
# n_components = 3  # Number of states in the HMM
# aa_model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag")
# iy_model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag")
# sh_model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag")

# # Train the HMM models
# aa_model.fit(aa_features)
# iy_model.fit(iy_features)
# sh_model.fit(sh_features)

# # Define a test sequence (MFCC features of a spoken phoneme)
# # test_sequence = np.array([[1.5, 2.5, 3.5], [3.5, 2.5, 1.5], [1.0, 2.0, 3.0]])
# test_sequence = np.array([[4.5, 3.0, 2.5], [3.0, 3.5, 0.5], [2.0, 1.0, 0.0]])

# # Use the models for phoneme recognition
# log_likelihood_aa = aa_model.score(test_sequence)
# log_likelihood_iy = iy_model.score(test_sequence)
# log_likelihood_sh = sh_model.score(test_sequence)

# # Determine the recognized phoneme
# phoneme_scores = {
#     "aa": log_likelihood_aa,
#     "iy": log_likelihood_iy,
#     "sh": log_likelihood_sh
# }
# recognized_phoneme = max(phoneme_scores, key=phoneme_scores.get)

# print(f"Recognized phoneme: {recognized_phoneme}")

import numpy as np
from hmmlearn import hmm

# Simulated MFCC features for three phonemes: "aa," "iy," and "sh"
data_aa = np.random.randn(10, 13)
data_iy = np.random.randn(10, 13)
data_sh = np.random.randn(10, 13)

# Create HMM models for "aa," "iy," and "sh"
n_components = 3  # Number of states in the HMM
aa_model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag")
iy_model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag")
sh_model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag")

# Train the HMM models
aa_model.fit(data_aa)
iy_model.fit(data_iy)
sh_model.fit(data_sh)

# Define a test sequence (actual MFCC features of a spoken phoneme)
test_sequence = np.array([[1.5, 2.5, 3.5, 4.0, 2.0, 1.0, 3.0, 2.0, 2.5, 3.0, 1.5, 2.0, 2.5]])

# Use the models for phoneme recognition
log_likelihood_aa = aa_model.score(test_sequence)
log_likelihood_iy = iy_model.score(test_sequence)
log_likelihood_sh = sh_model.score(test_sequence)

# Determine the recognized phoneme
phoneme_scores = {
    "aa": log_likelihood_aa,
    "iy": log_likelihood_iy,
    "sh": log_likelihood_sh
}
recognized_phoneme = max(phoneme_scores, key=phoneme_scores.get)

print(f"Recognized phoneme: {recognized_phoneme}")

