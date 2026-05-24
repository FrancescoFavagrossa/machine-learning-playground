# ML Project

Machine learning project for binary classification on the Tic-Tac-Toe dataset. The work compares several supervised learning approaches, evaluates them with hold-out validation, cross-validation, and Bayesian optimization, then summarizes the final model performances.

The final written report is available here: [ML_Report_finale finale.pdf](ML_Report_finale%20finale.pdf). The source notebook/report is [ML_project_final.qmd](ML_project_final.qmd).

## Project Workflow

![ML Project Workflow](README_assets/project_workflow.png)

## Objective

The project studies a binary classification problem where the target class is encoded as `-1` and `1`. The workflow includes:

- exploratory data analysis and correlation checks;
- train/test splitting with class-aware partitioning;
- feature scaling for distance-based models;
- model selection through hold-out validation, repeated cross-validation, and Bayesian optimization;
- LASSO-based variable selection;
- final comparison of all tested models.

## Models Compared

The following methods are implemented in the project:

- K-Nearest Neighbors using hold-out validation;
- K-Nearest Neighbors using repeated 10-fold cross-validation;
- K-Nearest Neighbors with Bayesian optimization;
- Support Vector Machine using 10-fold cross-validation;
- Support Vector Machine with Bayesian optimization;
- Gaussian Process classifier;
- Random Forest with Bayesian optimization;
- KNN with Bayesian optimization after LASSO variable selection;
- SVM with Bayesian optimization after LASSO variable selection.

## Final Results

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

Based on the final comparison, the strongest performances are obtained by the Gaussian Process, Random Forest with Bayesian optimization, and the LASSO-selected KNN/SVM variants. The simpler KNN and SVM configurations perform in the 0.64-0.70 accuracy range.

## Repository Structure

```text
.
├── Dataset/
│   ├── TicTacToe_dataset.RDS
│   ├── Clustering_dataset.RDS
│   ├── Saturn_dataset.RDS
│   └── syn_df.csv
├── Script/
│   ├── KNNBO.R
│   ├── SVMBO.R
│   ├── RFBO.R
│   ├── GP.R
│   └── XGBOOST.R
├── Plots/
│   └── Plots/
├── README_assets/
│   ├── accuracy_comparison.png
│   └── project_workflow.png
├── ML_project_final.qmd
├── ML_Report_finale finale.pdf
└── best_model.h5
```

## Main Dependencies

The analysis is written in R. The main packages used in `ML_project_final.qmd` are:

- `caret`
- `caTools`
- `corrplot`
- `DiceKriging`
- `dplyr`
- `e1071`
- `factoextra`
- `GauPro`
- `ggplot2`
- `glmnet`
- `gridExtra`
- `mlr`
- `mlrMBO`
- `parallelMap`
- `tidyverse`

## Reproducing the Report

1. Install R and the required packages listed above.
2. Install Quarto if you want to render the `.qmd` source.
3. Open or render [ML_project_final.qmd](ML_project_final.qmd).
4. If running the project on another machine, update the absolute dataset paths in the QMD file so they point to the local `Dataset/` folder.

The current final PDF can be used directly without rerunning the analysis.
