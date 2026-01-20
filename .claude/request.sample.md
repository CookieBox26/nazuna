### Request

This repository is for analyzing time-series forecasting models.
Please implement the following changes according to the steps below.

1. Add a parameter `tolerance` (default: 0) to the constructor of `MSELoss` in `nazuna/criteria.py`.
   If the error is smaller than this value, it should be treated as zero. Implement the logic accordingly.
2. Add unit tests for this change in `tests/test_criteria.py`.

### Steps

- Create a new branch named `feature/xxxxxx`, where `xxxxxx` is a short string consisting of alphanumeric characters and underscores.
- Commit the requested changes to this branch.
- Create `.claude/run.sh` (overwrite it if it already exists) to push the branch and open a pull request.
  Do not commit this file, and do not execute it.
- Print the following to standard output to notify me:
  - The PR title and PR description in English written in `.claude/run.sh`
  - A Japanese explanation or notes for the PR

```sh:.claude/run.sh
git push origin feature/xxxxxx
gh pr create --base main --head feature/xxxxxx --title 'PR title in English' --body 'PR description in English'

# Post-merge cleanup command
# git branch -d feature/xxxxxx
```

### Notes

- I will review the changes after the pull request is created. Do not ask me to review or confirm the changes before making the PR.
