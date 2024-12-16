# reddit-depression

Video link: https://drive.google.com/file/d/1Bj-wvV8gLKusv56S4eLSbgFwPnPnEn-F/view?usp=sharing

This project analyzes Reddit posts to identify symptoms of depression using topic modeling (LDA) and sentence embeddings (DistilRoBERTa). The analysis evaluates performance through metrics like AUC and F1 scores.

## Table of Contents

1. [Project Overview](#project-overview)  
2. [How to Run the Code](#how-to-run-the-code)  
3. [Known Issues and Limitations](#known-issues-and-limitations)  
4. [Extra Experiments](#extra-experiments)  

## Project Overview

The goal of this project is to detect and classify depressive symptoms from Reddit posts. We use:

- **LDA (Latent Dirichlet Allocation):** Topic modeling for depression-related subreddits.  
- **DistilRoBERTa:** A lightweight transformer for semantic embeddings and classification.


## How to Run the Code
1.	Open Reddit_Depression_Stencil.ipynb in Google Colab.
2.	Set up Google Drive and ensure all files (data and pickled models) are stored in your Drive folder as specified in `FILEPATH` and `FOLDER`.
3.	Install dependencies for data preperation
4.	Tokenize posts using the `tokenize()` function.
5.	Remove stopwords (Top 100 frequent words in the control dataset with `find_stopwords()`) from all the datasets.  
6.	Train LDA and evaluate with the  `lda_rf_cross_validation()` function.
7.	Generate DistilRoBERTa embeddings via `get_distilroberta_embeddings()` and evaluate with `drb_rf_cross_validation()`.
8. Optional F1 Evaluation: Run `rf_cross_validation_f1_cached()` for F1 score analysis.

## Known Issues and Limitations

**LDA Performance for Anger**:
 - The AUC score for "Anger" using LDA is 0.923 compared to the reference value of 0.794, and the F1 score is 0.678 compared to the reference value of 0.387. This discrepancy is quite odd because other symptoms align closely with reference values for both LDA and DistilRoBERTa.
 - I think potential issues are:
   1) **Overfitting**: The "Anger" dataset may have unique characteristics, such as a higher prevalence of specific keywords or more distinct topic distributions.
   2) **Higher Coherence**: The "Anger" dataset might have unique token distributions or higher coherence, so that it's easier for LDA to distinguish it from other symptoms.
   3) The reference used a slightly different subset of the dataset (e.g., extra filtering or different train/test splits).

**DistilRoBERTa Cross-Validation Runtime**:
 - Cross-validation with DistilRoBERTa can take up to **40 minutes**, potentially due to: 1) Batch processing of embeddings. 2) High-dimensional embeddings being passed to classifiers.

**DistilRoBERTa Limitation**:
   - Sentence truncation at 512 tokens so the model might lose some information when dealing with long posts, but my DistilRoBERTa scores (both AUC and F1) align closely with the reference data so I think it's largely fine.

## Extra Experiments
Optional F1 Evaluation: Feel free to run `rf_cross_validation_f1_cached()` and see my F1 score table! It might take around 40 minutes though. 
