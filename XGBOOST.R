library(mlrMBO) ## Bayesian Opt. Package 
library(parallelMap) ## Computational Opt.

ds <- readRDS("/Users/francescofavagrossa/Desktop/TicTacToe_dataset.RDS") ## load dataset
ds$class <- as.factor(ds$class)
# 1. CREATE TASK
task <- makeClassifTask(
  id = "binary_classification",
  data = ds,
  target = "class"
)

# 2. DEFINE LEARNER
learner <- makeLearner(
  "classif.xgboost",
  predict.type = "prob",
  par.vals = list(
    objective = "binary:logistic",
    eval_metric = "accc",
    nthread = 1  # set to 1 when using parallelMap
  )
)

# 3. PARAMETER SPACE
param_set <- makeParamSet(
  makeIntegerParam("nrounds", lower = 50, upper = 500),
  makeNumericParam("eta", lower = 0.01, upper = 0.3),
  makeIntegerParam("max_depth", lower = 3, upper = 10),
  makeNumericParam("subsample", lower = 0.5, upper = 1),
  makeNumericParam("colsample_bytree", lower = 0.5, upper = 1)
)

# 4. RESAMPLING
rdesc <- makeResampleDesc("CV", iters = 5, stratify = TRUE)

?makeResampleDesc
# 5. MBO CONTROL - CORRECTED!
# First create the MBO control object
mbo_ctrl <- makeMBOControl()
mbo_ctrl <- setMBOControlTermination(mbo_ctrl, iters = 30)
mbo_ctrl <- setMBOControlInfill(mbo_ctrl, crit = makeMBOInfillCritEI())

# Then wrap it in TuneControl
ctrl <- makeTuneControlMBO(mbo.control = mbo_ctrl)

# 6. PARALLEL SETUP
n_cores <- parallel::detectCores()
use_cores <- max(1, floor(n_cores * 0.75))
cat("Using", use_cores, "out of", n_cores, "cores\n")

set.seed(123)
parallelStartMulticore(cpus = use_cores, mc.set.seed = TRUE)

# 7. RUN OPTIMIZATION
result <- tuneParams(
  learner = learner,
  task = task,
  resampling = rdesc,
  measures = list(auc, acc),
  par.set = param_set,
  control = ctrl,  # Now using TuneControlMBO
  show.info = TRUE
)

parallelStop()

# 8. VIEW RESULTS
print(result)
cat("\nBest Hyperparameters:\n")
print(result$x)
cat("\nBest Performance:\n")
print(result$y)

# 9. TRAIN FINAL MODEL
final_learner <- setHyperPars(learner, par.vals = result$x)
final_model <- train(final_learner, task)

# 10. OPTIMIZATION PATH
opt_path <- as.data.frame(result$opt.path)
head(opt_path)

# 11. PLOT PROGRESS
library(ggplot2)
ggplot(opt_path, aes(x = seq_along(auc.test.mean), y = auc.test.mean)) +
  geom_line(color = "blue", alpha = 0.6) +
  geom_point(color = "darkblue", size = 2) +
  geom_hline(yintercept = max(opt_path$auc.test.mean), 
             linetype = "dashed", color = "red") +
  labs(title = "Bayesian Optimization Progress",
       x = "Iteration", y = "AUC (CV)") +
  theme_minimal()