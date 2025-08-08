chore(pre-commit): fix mdformat hook by adding gfm plugin and pinning markdown-it-py\<3.0.0

## Explanation

The `mdformat` pre-commit hook was failing on pre-commit.ci with:

```
KeyError: 'Parser rule not found: linkify'
```

This originates from the GFM autolink plugin expecting the `linkify` rule, which was not registered due to a plugin / dependency mismatch after upstream `markdown-it-py` 3.x changes. The hook environment (without explicit GFM + version pin) pulled a newer combination that broke the plugin ordering.

## Root Cause

- `mdformat` alone does not bundle GitHub-Flavored Markdown extensions.
- `mdformat-gfm` must be explicitly added to `additional_dependencies` to enable GFM extensions (including autolink behavior).
- Newer `markdown-it-py` releases changed internals causing the `gfm_autolink` plugin to try to insert before a rule that no longer exists in that form.
- Without a version pin, pre-commit.ci resolved an incompatible `markdown-it-py` version, triggering the KeyError.

## Changes

- Added: `mdformat-gfm==0.3.5` to `additional_dependencies`.
- Pinned: `markdown-it-py<3.0.0` to ensure compatibility with the current `mdformat-gfm` plugin.
- Left existing `mdformat_frontmatter` dependency intact.
- Verified locally in a clean virtual environment that the hook now passes.

## Reproduction (Before Fix)

(Reference prior failed CI run logs showing the KeyError; can be reproduced by removing the added dependencies and running:)

```
python3 -m venv .venv
source .venv/bin/activate
pip install pre-commit mdformat
pre-commit run --all-files mdformat
```

Result before fix: traceback with `KeyError: 'Parser rule not found: linkify'`.

## Verification (After Fix)

```
python3 -m venv .venv
source .venv/bin/activate
pip install pre-commit
pre-commit run --all-files mdformat
```

Result: `mdformat....................................Passed`

Also validated full `pre-commit run --all-files` succeeds for the mdformat stage locally.

## Notes

- No existing issue number referenced (CI unblock maintenance task).
- This is isolated to formatting infrastructure; no runtime code paths touched.
- Separately, docstring formatting remains covered by `black`; `isort` continues to manage import style.
- A future follow-up can relax the `markdown-it-py` pin once upstream `mdformat-gfm` declares compatibility with 3.x.

## Future Cleanup

Periodically test removing the pin:

```
pre-commit autoupdate
# remove markdown-it-py<3.0.0 from additional_dependencies
pre-commit run --all-files mdformat
```

If it still passes, the pin can be dropped.

## Checklist

- [x] Change limited to developer / CI tooling
- [x] Local verification of hook success
- [x] Clear explanation and reproduction included
- [x] No docs or API changes required
