git push origin feature/mseloss_tolerance
gh pr create --base main --head feature/mseloss_tolerance --title 'Add tolerance parameter to MSELoss' --body 'Add tolerance parameter to MSELoss class that treats errors below a specified threshold as zero.

## Changes
- Added `tolerance` parameter to `MSELoss.__init__` with default value of 0
- Modified `calc_loss` method to set differences below tolerance to zero before squaring
- Added unit tests for the new tolerance functionality

## Usage
```python
criterion = MSELoss.create(device, n_channel=3, pred_len=4, tolerance=1.5)
```
Errors with absolute value less than 1.5 will be treated as zero.'
