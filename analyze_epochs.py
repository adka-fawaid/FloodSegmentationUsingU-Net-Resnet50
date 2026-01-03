import pandas as pd

df = pd.read_csv('Results/OFAT_resnet50/epochs.csv')

print("=" * 70)
print("ANALISIS HASIL OFAT - EPOCHS OPTIMIZATION")
print("=" * 70)

# Group by param_value (epochs)
for pv in sorted(df['param_value'].unique()):
    subset = df[df['param_value'] == pv]
    
    # Final epoch result
    final = subset.iloc[-1]
    final_iou = final['val_iou']
    final_dice = final['val_dice']
    final_score = (final_iou + final_dice) / 2
    
    # Best epoch result
    best_idx = subset[['val_iou', 'val_dice']].mean(axis=1).idxmax()
    best = subset.loc[best_idx]
    best_iou = best['val_iou']
    best_dice = best['val_dice']
    best_score = (best_iou + best_dice) / 2
    best_epoch = best['epoch']
    
    print(f"\n{int(pv)} EPOCHS:")
    print(f"  Final (epoch {int(final['epoch'])}):")
    print(f"    IoU:   {final_iou:.4f}")
    print(f"    Dice:  {final_dice:.4f}")
    print(f"    Score: {final_score:.4f}")
    print(f"  Best (epoch {int(best_epoch)}):")
    print(f"    IoU:   {best_iou:.4f}")
    print(f"    Dice:  {best_dice:.4f}")
    print(f"    Score: {best_score:.4f}")

print("\n" + "=" * 70)
print("REKOMENDASI UNTUK TABEL:")
print("=" * 70)

# Find best overall based on final score
results = []
for pv in df['param_value'].unique():
    subset = df[df['param_value'] == pv]
    final = subset.iloc[-1]
    final_score = (final['val_iou'] + final['val_dice']) / 2
    results.append({
        'epochs': int(pv),
        'final_iou': final['val_iou'],
        'final_dice': final['val_dice'],
        'final_score': final_score
    })

results_df = pd.DataFrame(results).sort_values('final_score', ascending=False)

print("\nRanking berdasarkan Final Score (IoU + Dice)/2:")
for idx, row in results_df.iterrows():
    print(f"{int(row['epochs']):3d} epochs: IoU={row['final_iou']:.4f}, Dice={row['final_dice']:.4f}, Score={row['final_score']:.4f}")

best = results_df.iloc[0]
print(f"\n✅ TERBAIK: {int(best['epochs'])} epochs")
print(f"   IoU:  {best['final_iou']:.4f}")
print(f"   Dice: {best['final_dice']:.4f}")
