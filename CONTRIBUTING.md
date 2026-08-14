# Contributing to wdflow

Changes reach `master` through pull requests only, and a pull request merges
only once CI is green. Nobody pushes to `master` directly, including the
maintainer.

## Making a change

```bash
git checkout -b short-description-of-the-change
# work, commit
git push -u origin short-description-of-the-change
gh pr create --fill        # or open the PR from the GitHub web interface
```

Run the tests before you push. Most of them need nothing compiled:

```bash
pip install -e ".[dev]"
pytest tests \
  --ignore=tests/test_golden_output.py \
  --ignore=tests/test_reconstruction.py \
  --ignore=tests/test_zero_phase_whitening.py \
  --ignore=tests/test_gnn.py \
  --ignore=tests/test_mock_dataset.py
```

The three ignored pipeline tests drive real trigger generation and need
[p4TSA](https://github.com/elenacuoco/p4TSA) built from source; CI runs them for
you. `test_gnn.py` needs the `gnn` extra and `test_mock_dataset.py` needs
`pycbc`.

## What CI checks

Every pull request runs three jobs, all of which must pass:

| Job | What it does |
|-----|--------------|
| `Analysis layer (Python 3.10 / 3.11 / 3.12)` | The analysis suite on a plain pip install, on every supported Python. |
| `Full suite (p4TSA built from source)` | Builds the C++ core and runs trigger generation end to end, including the golden-output regression. |
| `Docs build check` | Sphinx with `-W`, so a documentation warning fails the build. |

If you change trigger generation and the golden-output test fails, that is the
test doing its job: it pins exact numerics on a fixed synthetic frame. Either
the change was not meant to move them, or it was — in which case regenerate
`tests/fixtures/golden_triggers.parquet`, and say in the pull request which columns
moved and why.

## House style

- Functions do one thing, and the docstring says what they do: what it does,
  then `:param:` / `:return:`. No explanatory comments inside the body.
- No branching for one particular input, and no documenting around a specific
  data set or a specific bug.
- English everywhere, in code, docstrings and notebooks.
- The detection statistic is `EnWDF` throughout. Foreground and background must
  be ranked on the same quantity for a false-alarm probability to mean
  anything, so a new statistic means regenerating the background too.

## For the maintainer: enforcing this

The rule above is a repository setting, not something in this file. Enable it
once, on GitHub, under *Settings → Branches → Add branch ruleset* for `master`:
require a pull request before merging, and require these status checks to pass:

```
Analysis layer (Python 3.10)
Analysis layer (Python 3.11)
Analysis layer (Python 3.12)
Full suite (p4TSA built from source)
Docs build check
```

Tick *Do not allow bypassing the above settings* so it applies to
administrators too — otherwise the maintainer keeps a direct-push path and the
rule is advisory.

The same thing from the command line:

```bash
gh api -X PUT repos/elenacuoco/wdflow/branches/master/protection \
  -F required_pull_request_reviews[required_approving_review_count]=0 \
  -F required_status_checks[strict]=true \
  -F 'required_status_checks[contexts][]=Analysis layer (Python 3.10)' \
  -F 'required_status_checks[contexts][]=Analysis layer (Python 3.11)' \
  -F 'required_status_checks[contexts][]=Analysis layer (Python 3.12)' \
  -F 'required_status_checks[contexts][]=Full suite (p4TSA built from source)' \
  -F 'required_status_checks[contexts][]=Docs build check' \
  -F enforce_admins=true \
  -F restrictions=
```

A status check can only be marked as required after it has reported once, so
push the workflow first and let it run one pull request before adding the rule.
