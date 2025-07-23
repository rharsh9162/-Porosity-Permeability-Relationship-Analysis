import pandas as pd
import numpy as np

# --- Parameters for Data Generation (tuned to resemble the image) ---
num_samples = 3000  # A large dataset
np.random.seed(42) # for reproducibility

# Porosity range (X-axis)
min_porosity = 2.0
max_porosity = 28.0

# Define the underlying log-linear relationship for permeability (Y-axis on log scale)
# This is derived by visually estimating the red line in the log-permeability plot:
# ln(Permeability) = slope * Porosity + intercept
# From the image:
# At Porosity ~5%, Permeability is ~1 mD (ln(1) = 0)
# At Porosity ~25%, Permeability is ~1000 mD (ln(1000) approx 6.907)
# Slope = (6.907 - 0) / (25 - 5) = 6.907 / 20 = 0.34535
# Intercept = 0 - (0.34535 * 5) = -1.72675
linear_slope = 0.34535
linear_intercept = -1.72675

# Noise level for the log-permeability values.
# Higher std_dev_noise means more scatter on the log plot.
std_dev_noise = 1.0 # This creates noticeable scatter, resembling the image

# --- Generate Data ---

# 1. Generate Porosity values
porosity = np.random.uniform(min_porosity, max_porosity, num_samples)
porosity = np.round(porosity, 2) # Round to 2 decimal places for realism

# 2. Calculate ideal Log Permeability based on the linear model
log_permeability_ideal = linear_slope * porosity + linear_intercept

# 3. Add Gaussian noise to Log Permeability
# This noise will look symmetrical on the log scale, asymmetrical on linear scale
log_permeability_noisy = log_permeability_ideal + np.random.normal(0, std_dev_noise, num_samples)

# 4. Convert Log Permeability back to linear Permeability
permeability = np.exp(log_permeability_noisy)

# 5. Ensure permeability values are positive (should be due to exp, but good check)
# And set a minimum sensible value, as permeability cannot be zero or negative
min_sensible_permeability = 0.001
permeability[permeability < min_sensible_permeability] = min_sensible_permeability
permeability = np.round(permeability, 3) # Round permeability to 3 decimal places

# --- Create DataFrame ---
data = pd.DataFrame({
    'Porosity': porosity,
    'Permeability': permeability
})

# Sort by Porosity for better visual representation if plotted directly without scatter
data = data.sort_values(by='Porosity').reset_index(drop=True)

# --- Save to CSV ---
csv_file_name = 'porosity_permeability_dataset.csv'
data.to_csv(csv_file_name, index=False)

print(f"Successfully generated '{csv_file_name}' with {num_samples} data points.")
print("\nFirst 5 rows of the generated data:")
print(data.head())
print(f"\nPermeability Min: {data['Permeability'].min():.4f}, Max: {data['Permeability'].max():.4f}")
print(f"Porosity Min: {data['Porosity'].min():.2f}, Max: {data['Porosity'].max():.2f}")