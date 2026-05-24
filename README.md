# ML Project Quarto Report

This README describes the Quarto document [ML_project_final.qmd](ML_project_final.qmd), which contains the full machine learning analysis and final model comparison for the Tic-Tac-Toe binary classification project.




<img width="357" height="293" alt="Screenshot 2026-05-24 at 22 57 30" src="https://github.com/user-attachments/assets/3e3c0dda-83c3-462b-a9c8-b68539a7525f" />


## What The Quarto Document Contains

[ML_project_final.qmd](ML_project_final.qmd) is organized as a complete analysis report. It includes:

- exploratory data analysis and feature correlation plots;
- preprocessing and train/test split setup;
- K-Nearest Neighbors models with hold-out validation, cross-validation, and Bayesian optimization;
- Support Vector Machine models with cross-validation and Bayesian optimization;
- Gaussian Process classification;
- Random Forest tuning with Bayesian optimization;
- LASSO variable selection;
- final comparison of all tested models.

## Final Accuracy Plot

The final section of the Quarto document compares all proposed models using test accuracy.

![Accuracy Comparison Across Models](README_assets/accuracy_comparison.png)

| Model | Accuracy |
|---|---:|
| KNN Hold Out | 0.640 |
| KNN Cross Validation | 0.700 |
| KNN Bayesian Optimization | 0.690 |
| SVM Cross Validation | 0.685 |
| SVM Bayesian Optimization | 0.671 |
| Gaussian Process | 1.000 |
| Random Forest Bayesian Optimization | 1.000 |
| KNN Bayesian Optimization with LASSO | 1.000 |
| SVM Bayesian Optimization with LASSO | 1.000 |

## Required R Packages

The Quarto file uses the following main R packages:

```r
library(caTools)
library(caret)
library(corrplot)
library(DiceKriging)
library(dplyr)
library(e1071)
library(factoextra)
library(farff)
library(GauPro)
library(ggplot2)
library(gridExtra)
library(mlr)
library(mlrMBO)
library(parallelMap)
library(tibble)
library(tidyverse)
library(glmnet)
library(tidyr)
```

## Rendering The Quarto Document

To render the report, install Quarto and run:

```bash
quarto render ML_project_final.qmd
```

The current final exported report is available as [ML_Report_finale finale.pdf](ML_Report_finale%20finale.pdf).

## Note On File Paths

The Quarto document currently uses absolute local paths for the dataset, for example:

```r
readRDS("/Users/francescofavagrossa/Desktop/Machine Learning/ML Project/Dataset/TicTacToe_dataset.RDS")
```

If the project is moved to another machine or folder, update those paths to match the new location.
