import pandas as pd
import numpy as np

# Load rerun_seeds_summary.csv
df = pd.read_csv('Results/ResNet50_rerun_seeds/best_config/rerun_seeds_summary.csv')

print("=" * 80)
print("STATISTIK RERUN DENGAN MULTIPLE SEEDS - BEST CONFIG")
print("=" * 80)

print("\nData Raw:")
print(df[['seed', 'test_iou', 'test_dice', 'val_iou', 'val_dice']].to_string(index=False))

# Calculate mean and std
metrics = ['test_iou', 'test_dice', 'val_iou', 'val_dice']
stats = {}

print("\n" + "=" * 80)
print("MEAN & STD untuk setiap metrik:")
print("=" * 80)

for metric in metrics:
    values = df[metric].values
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)  # Sample std
    stats[metric] = {'mean': mean_val, 'std': std_val}
    print(f"\n{metric.upper()}:")
    print(f"  Mean: {mean_val:.4f}")
    print(f"  Std:  {std_val:.4f}")
    print(f"  Format: {mean_val:.4f} ± {std_val:.4f}")

print("\n" + "=" * 80)
print("RINGKASAN UNTUK TABEL:")
print("=" * 80)

print("\n| Metric     | Seed 0  | Seed 42 | Seed 123| Mean    | Std     |")
print("|------------|---------|---------|---------|---------|---------|")
for metric in metrics:
    metric_name = metric.replace('_', ' ').title()
    vals = df[metric].values
    print(f"| {metric_name:10s} | {vals[0]:.4f}  | {vals[1]:.4f}  | {vals[2]:.4f}  | {stats[metric]['mean']:.4f}  | {stats[metric]['std']:.4f}  |")

print("\n" + "=" * 80)
print("FORMAT UNTUK JURNAL:")
print("=" * 80)
print(f"\nTest IoU:  {stats['test_iou']['mean']:.4f} ± {stats['test_iou']['std']:.4f}")
print(f"Test Dice: {stats['test_dice']['mean']:.4f} ± {stats['test_dice']['std']:.4f}")
print(f"Val IoU:   {stats['val_iou']['mean']:.4f} ± {stats['val_iou']['std']:.4f}")
print(f"Val Dice:  {stats['val_dice']['mean']:.4f} ± {stats['val_dice']['std']:.4f}")

# Save statistics to CSV
stats_df = pd.DataFrame({
    'metric': metrics,
    'mean': [stats[m]['mean'] for m in metrics],
    'std': [stats[m]['std'] for m in metrics],
    'seed_0': df[metrics].iloc[0].values,
    'seed_42': df[metrics].iloc[1].values,
    'seed_123': df[metrics].iloc[2].values
})

output_file = 'Results/ResNet50_rerun_seeds/best_config/seed_statistics.csv'
stats_df.to_csv(output_file, index=False)
print(f"\n✅ Statistik disimpan ke: {output_file}")
