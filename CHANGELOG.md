# Changelog

## [2026-01-02] - OFAT Bug Fixes & Improvements

### Fixed
- **OFAT Logic Bugs**:
  - Fixed `check_phase_completion`: Now uses final epoch + score instead of max IoU/Dice separately
  - Fixed `optimize_parameter` resume logic: Uses final epoch instead of separate max values
  - Fixed return value bug: Returns `(best_value, all_results)` instead of `(best_value, best_iou)`
  - Fixed Phase 4 best_val_iou calculation: Now correctly takes max from selected best_params['epochs']
  - Fixed Phase 4 resume: Properly reads CSV to get best val_iou instead of using final_iou

### Improved
- **Consistency**: All OFAT phases now use consistent selection logic (final epoch + score)
- **best_config.yaml Generation**: Now accurately reflects the best parameters and their performance
- **Documentation**: Added comprehensive README.md with usage instructions
- **Reproducibility**: Fixed multi-seed runs to separate results by config name

### Added
- Validation scripts (`validate_ofat_logic.py`, `check_all_ofat.py`) for testing OFAT correctness
- Comprehensive `.gitignore` for ML projects
- `requirements.txt` with all dependencies
- Detailed README.md with project overview and usage

### Technical Details

**Bug Impact:**
- Previous OFAT runs had inconsistent parameter selection due to:
  1. Comparing final_score with best_iou (different metrics)
  2. Taking max IoU and max Dice from different epochs
  3. Wrong best_val_iou in best_config.yaml (from wrong phase)

**Solution:**
- All selection based on final epoch performance (consistent with actual model behavior)
- Score calculated as average of IoU and Dice from same epoch
- best_val_iou correctly represents the best IoU achieved during training with selected parameters

### Notes
- **Action Required**: Re-run OFAT for Baseline and ResNet50 to get corrected results
- All fixes are backward compatible with existing CSV files (can resume)
- Script will automatically use correct logic for future runs
