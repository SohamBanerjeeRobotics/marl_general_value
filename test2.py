# Step 2 — Pick the best model (smallest distance = best performance)
# The filename contains the distance score, e.g:
# epoch-2800-distance-0.84-return-12.3.eqx  ← smaller distance is better

import glob, os, re

model_files = glob.glob("models/**/*.eqx", recursive=True)

best_file = None
best_dist = float("inf")
for f in model_files:
    # extract distance from filename
    match = re.search(r'distance-([\d.]+)', f)
    if match:
        dist = float(match.group(1))
        if dist < best_dist:
            best_dist = dist
            best_file = f

print(f"✅ Best model: {best_file}")
print(f"   Distance:   {best_dist:.4f}  (lower = better)")

# Copy it to a convenient path
import shutil
shutil.copy(best_file, "models/best_model.eqx")
print(f"\n✅ Copied to: models/best_model.eqx")
