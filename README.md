# SOAK: Same/Other/All K-fold Cross-Validation
SOAK is designed to estimate the **similarity of patterns** found across different subsets of a dataset. It extends traditional K-fold cross-validation with "Same," "Other," and "All" splitting strategies to provide a robust measure of pattern similarity.
- Same/Other/All Splits:
  - Same: Subsets having observations in the same group
  - Other: Subsets having observations in the other group
  - All: Both Same and Other.

- Evaluation models:
   - featureless
   - cv_glmnet

- Languague: Python

# TODO
- Find more regression datasets:
   - source: uci repo or openml
   - having different categories (such as gender)
   - prioritize datasets having many citations
