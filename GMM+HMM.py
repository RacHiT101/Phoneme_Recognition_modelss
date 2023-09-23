import numpy as np
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm

# Simulated phoneme data (replace with your dataset)
phoneme_data = {
    "aa": np.random.randn(100, 13),  # Example features for "aa"
    "iy": np.random.randn(100, 13),  # Example features for "iy"
    "sh": np.random.randn(100, 13),  # Example features for "sh"
}

# Create Gaussian Mixture Models (GMMs) for each phoneme
gmm_aa = GaussianMixture(n_components=3, covariance_type='diag')
gmm_iy = GaussianMixture(n_components=3, covariance_type='diag')
gmm_sh = GaussianMixture(n_components=3, covariance_type='diag')

# Fit the GMMs to the phoneme data
gmm_aa.fit(phoneme_data["aa"])
gmm_iy.fit(phoneme_data["iy"])
gmm_sh.fit(phoneme_data["sh"])

# Create HMM models for phoneme recognition
n_states = 5  # Number of states in the HMM
aa_model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag")
iy_model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag")
sh_model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag")

# Train the HMM models (use GMM output as observations)
aa_model.fit(gmm_aa.predict(phoneme_data["aa"]).reshape(-1, 1))
iy_model.fit(gmm_iy.predict(phoneme_data["iy"]).reshape(-1, 1))
sh_model.fit(gmm_sh.predict(phoneme_data["sh"]).reshape(-1, 1))

# Define a test sequence (replace with your test data)
test_sequence = np.random.randn(50, 13)

# Use the HMM models for phoneme recognition
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
