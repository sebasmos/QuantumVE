import numpy as np
import os

# Path to your .npz file
npz_path = "../data/mnist/distilled_data.npz"

# Load the NPZ archive
data = np.load(npz_path)

# Print available keys
print("📂 Loaded keys in distilled_data.npz:")
for key in data.files:
    print(f" - {key}")

print("\n📊 Shape summary:")
for key in data.files:
    print(f"{key:20s} shape = {data[key].shape}")

# Assertions
assert 'full_flattened' in data and 'full_labels' in data, "Missing full dataset in .npz"
assert data['full_flattened'].shape[0] == 70000, "Expected 70,000 flattened images"
assert data['full_labels'].shape[0] == 70000, "Expected 70,000 labels"

print("\n✅ Assertions passed: full dataset contains exactly 70,000 samples.")

# Optionally print sample counts for train/test
print(f"\n🧪 Training samples: {data['X_train'].shape[0]}")
print(f"🧪 Testing samples : {data['X_test'].shape[0]}")