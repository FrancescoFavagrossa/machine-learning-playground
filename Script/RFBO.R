rm(list = ls()) ; graphics.off()
set.seed(123)
library(mlr)
library(mlrMBO)
library(parallelMap)
library(caret)

# 1. IMPORT DATA AND DATA PARTITION

data <- readRDS("/Users/francescofavagrossa/Desktop/ML Project/Dataset/TicTacToe_dataset.RDS")
data$class <- as.factor(data$class)
train <- createDataPartition(data$class, p=0.8, list=FALSE)
ds <- data[train,]
test <- data[-train, -ncol(data)]
test_label <- data[-train, ]$class


# 2. CREATE TASK AND LEARNER

task <- makeClassifTask(
  id = "binary_classification",
  data = ds,
  target = "class"
)

learner <- makeLearner(
  "classif.randomForest",
  predict.type = "prob"
)

# 3. PARAMETER SPACE FOR RF

param_set <- makeParamSet(
  makeIntegerParam("ntree", lower = 100, upper = 1000),
  makeIntegerParam("mtry", lower = 1, upper = 12),
  makeIntegerParam("nodesize", lower = 1, upper = 20),
  makeIntegerParam("maxnodes", lower = 10, upper = 100)
)

# 4. RESAMPLING -CV
rdesc <- makeResampleDesc("CV", iters = 10, stratify = TRUE)

# 5. MBO CONTROL
mbo_ctrl <- makeMBOControl()
mbo_ctrl <- setMBOControlTermination(mbo_ctrl, iters = 30)
mbo_ctrl <- setMBOControlInfill(mbo_ctrl, crit = makeMBOInfillCritEI())

# Then wrap it in TuneControl
ctrl <- makeTuneControlMBO(
  mbo.control = mbo_ctrl,
  mbo.design = generateDesign(n = 10, par.set = param_set)
)

# 6. PARALLEL SETUP JUST FOR BETTER COMPUTATION
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
  measures = list(acc),
  par.set = param_set,
  control = ctrl,
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
final_model <- mlr::train(final_learner, task)

# 10. OPTIMIZATION PATH
opt_path <- as.data.frame(result$opt.path)

# 11. PLOT PROGRESS

# Plot accuracy progress
p1 <- ggplot(opt_path, aes(x = seq_along(acc.test.mean), y = acc.test.mean)) +
  geom_line(color = "blue", alpha = 0.6) +
  geom_point(color = "darkblue", size = 3) +
  geom_hline(yintercept = max(opt_path$acc.test.mean), 
             linetype = "dashed", color = "red") +
  labs(title = "Random Forest Bayesian Optimization Progress",
       x = "Iteration", y = "Accuracy") +
  theme_minimal()
print(p1)

# Plot parameter space exploration (ntree vs mtry)
p2 <- ggplot(opt_path, aes(x = ntree, y = mtry)) +
  geom_point(aes(color = acc.test.mean, size = acc.test.mean), alpha = 0.7) +
  scale_color_gradient(low = "red", high = "green") +
  labs(title = "Random Forest Parameter Space Exploration",
       x = "Number of Trees", y = "mtry",
       color = "Accuracy", size = "Accuracy") +
  theme_minimal()
print(p2)

# Plot nodesize effect
p3 <- ggplot(opt_path, aes(x = nodesize, y = acc.test.mean)) +
  geom_point(aes(color = mtry), size = 3) +
  geom_smooth(method = "loess", se = TRUE, color = "blue") +
  labs(title = "Effect of Node Size on Accuracy",
       x = "Node Size", y = "Accuracy") +
  theme_minimal()
print(p3)

# 12. FINAL MODEL PERFORMANCE

preds <- predict(final_model, newdata = test)$data$response
df_preds <- data.frame(Actual = test_label, Predicted = preds)

conf_mat <- table(Predicted = df_preds$Predicted, Actual = df_preds$Actual)

# Convert to data frame for ggplot
conf_mat_df <- as.data.frame(conf_mat)

# Confusion Matrix plot
p4 <- ggplot(conf_mat_df, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = Freq), size = 8, color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  coord_fixed() +
  labs(title = "Confusion Matrix for RF - BO (Test Set)",
       x = "Classe Predetta",
       y = "Classe Reale") +
  theme_minimal(base_size = 14) +
  theme(panel.grid = element_blank())

print(p4)

