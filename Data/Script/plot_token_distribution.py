import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

processed_dir = Path(__file__).parent.parent / 'Processed'

with open(processed_dir / 'merged_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

token_counts = [len(record['paragraph'].split()) for record in data]

mean_val = np.mean(token_counts)
median_val = np.median(token_counts)

print(f"Total procedures: {len(token_counts)}")
print(f"Mean tokens: {mean_val:.1f}")
print(f"Median tokens: {median_val:.1f}")
print(f"Std: {np.std(token_counts):.1f}")
print(f"Min: {np.min(token_counts)} | Max: {np.max(token_counts)}")
print(f"25th percentile: {np.percentile(token_counts, 25):.0f}")
print(f"75th percentile: {np.percentile(token_counts, 75):.0f}")
over_500 = sum(1 for c in token_counts if c > 500)
print(f"Procedures with >500 words: {over_500} ({over_500/len(token_counts)*100:.1f}%)")

plt.figure(figsize=(8, 5))
plt.hist(token_counts, bins=50)
plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.0f}')
plt.axvline(median_val, color='orange', linestyle='--', label=f'Median: {median_val:.0f}')
plt.xlabel('Token count (words)')
plt.ylabel('Frequency')
plt.title('Token distribution per procedure')
plt.legend()
plt.savefig(processed_dir / 'token_distribution.png', dpi=150)
plt.show()
print(f"Plot saved to {processed_dir / 'token_distribution.png'}")
