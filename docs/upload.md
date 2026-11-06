# Private repository upload checklist

1. Confirm the organization, repository visibility and internal code/data-sharing
   approvals. This package does not grant an open-source license.
2. Extract the ZIP. Upload the contents of its `qpm-sentinel/` folder, including
   `.gitignore`; do not upload the ZIP as the only repository file.
3. Keep raw archives, generated reports, private studies, reference/model assets
   and machine-specific bank pointers outside Git. `.gitignore` is a safeguard,
   not a substitute for review.
4. Install the tested dependencies from an approved package index. Run:

   ```bash
   python tools/run_tests.py
   python examples/run_synthetic_demo.py --output outputs/demo
   python tools/validate_package.py
   ```

5. Review the README figures and interpretation limits. They are frozen study
   summaries; the demo is synthetic. Do not relabel them as prospective clinical,
   calibrated-volume or causal validation.
6. Use your existing authenticated Git setup. Do not put an access token in a
   remote URL, README, script, configuration file or shell history.

## Optional Git command sequence

Replace `YOUR_PRIVATE_REPOSITORY_URL` with your approved repository URL. Review
the staged file list before committing or pushing.

```bash
git init
git branch -M main
git add .
git status --short
git diff --cached --stat
git commit -m "Add QPM Sentinel source, tests and visual documentation"
git remote add origin YOUR_PRIVATE_REPOSITORY_URL
git push -u origin main
```

These are instructions only. The packaging task has not created a remote
repository, committed changes or uploaded any material.

For future code packages, run `python tools/package_release.py --output NEW_NAME.zip`
after the tests and package checker pass. The packager refuses to silently
overwrite an existing output, verifies ZIP integrity and records SHA-256 hashes.

