# Project rules

## Writing rules

- Never use em dashes or en dashes in code comments, docstrings, or content strings.
  Use hyphens or rewrite the sentence instead.
  - Wrong: `# smoothing too aggressively -- when temp rises`
  - Right: `# smoothing too aggressively. When temp rises`
- All UI text in sentence case. Never ALL CAPS in component strings except intentional display headings.

## Git commit rules

- NEVER include `Co-Authored-By` in commit messages.
- NEVER include "Generated with Claude Code" or any AI trailer in commits.
- NEVER use `git add .` or `git add -A`. Always stage explicit file paths.
- NEVER force push to any remote.
- NEVER amend already-pushed commits.
- Commit message format: `type(scope): short description`
  - Examples: `feat(data): add tourist sales floor to outerwear generation`
  - Examples: `fix(dashboard): correct stale cache after data reload`
  - Types: feat, fix, style, refactor, chore, docs

## Python style

- No inline comments that state what the code does. Only add a comment when the WHY is non-obvious.
- No multi-line docstrings on internal functions. One short line max.
- No error handling for scenarios that cannot happen. Trust SQLAlchemy and pandas guarantees at the internal boundary; only validate at system entry points (user inputs, external files).

## Environment

- All scripts must be run with `PYTHONPATH=.` from the project root.
- Single Anaconda Python 3.12 environment for data generation, training, and serving.
- XGBoost models saved in `.ubj` native format, not pickle.
- SHAP TreeExplainer recreated at runtime from the loaded model; never pickled separately.
- Database: `data/uniqlo.db` (SQLite). All dashboard queries use `pd.read_sql()`. No CSV reads at runtime.
