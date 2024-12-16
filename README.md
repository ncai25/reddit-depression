# reddit-depression

## Project Overview

To Detect and classify depressive symptoms from Reddit posts, I use:
- **LDA (Latent Dirichlet Allocation):** Topic modeling for depression-related subreddits.  
- **DistilRoBERTa:** A lightweight transformer for semantic embeddings and classification.

## Table of Contents

1. [How to Run the Code](#how-to-run-the-code) 
2. [Results](#results)
3. [Known Issues and Limitations](#known-issues-and-limitations)  
4. [Extra Experiments](#extra-experiments)  

## How to Run the Code
1. Install dependencies for data preparation
2.	Tokenize posts using `tokenize()` + Remove stopwords (Top 100 frequent words in the control dataset with `find_stopwords()`)
3.	Train LDA and evaluate with the  `lda_rf_cross_validation()` function.
4.	Generate DistilRoBERTa embeddings via `get_distilroberta_embeddings()` and evaluate with `drb_rf_cross_validation()`.
5. Optional F1 Evaluation: Run `rf_cross_validation_f1_cached()` for F1 score analysis.

## Results
--- Final AUC Results with References ---

| Symptom              | LDA Test AUC | LDA Ref AUC | DRB Test AUC | DRB Ref AUC |
|----------------------|--------------|-------------|--------------|-------------|
| Anger               | 0.922        | 0.794       | 0.944        | 0.928       |
| Anhedonia           | 0.957        | 0.906       | 0.959        | 0.956       |
| Anxiety             | 0.938        | 0.837       | 0.956        | 0.952       |
| Disordered Eating   | 0.965        | 0.905       | 0.955        | 0.952       |
| Loneliness          | 0.845        | 0.806       | 0.916        | 0.907       |
| Sad Mood            | 0.848        | 0.788       | 0.935        | 0.919       |
| Self-Loathing       | 0.868        | 0.815       | 0.934        | 0.922       |
| Sleep Problems      | 0.976        | 0.909       | 0.958        | 0.956       |
| Somatic Complaints  | 0.915        | 0.880       | 0.931        | 0.925       |
| Worthlessness       | 0.757        | 0.700       | 0.924        | 0.897       |

## Known Issues and Limitations

1. **LDA Performance for Anger**
 - AUC: 0.923 (vs. reference 0.794), F1: 0.678 (vs. reference 0.387).
 - Unlike other symptoms, LDA significantly outperforms the reference. Possible reasons:
   1) Overfitting: Unique features or keyword prevalence in the “Anger” dataset.
   2) Higher Coherence: More distinct token/topic distributions for “Anger.”
   3) Dataset Variation: Reference may use a slightly different subset or train/test split.

2. **DistilRoBERTa Cross-Validation Runtime**:
 - Cross-validation takes ~40 minutes, likely due to:
    1) Batch processing of embeddings.
    2) High-dimensional embeddings being passed to classifiers.

## Extra Experiments
Optional F1 Evaluation: Feel free to run `rf_cross_validation_f1_cached()` and see my F1 score table! It might take around 40 minutes though. 
