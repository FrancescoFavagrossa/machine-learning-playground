rm(list = ls()) ; graphics.off()
library(mlrMBO)
library(rgenoud)
library(randomForest)
library(caret)

# 1. IMPORT DATA AND DATA PARTITION

data <- readRDS("/Users/francescofavagrossa/Desktop/ML Project/Dataset/TicTacToe_dataset.RDS")
data$class <- as.factor(data$class)
train <- createDataPartition(data$class, p=0.8, list=FALSE)
ds <- data[train,]
test <- data[-train, -ncol(data)]
test_label <- data[-train, ]$class

# 2. CREATE TASK AND LEARNER

task <- makeClassifTask(data = ds, target = "class")

learner <- makeLearner(
  "classif.svm",
  predict.type = "prob"
)

# 3. PARAMETER SPACE 

par.set <- makeParamSet(makeNumericParam("cost", lower = -2, upper = 2, trafo = function(x) 10 ^x),
                        makeDiscreteParam("kernel", values = c("linear", "radial")),
                        makeNumericParam("gamma", lower = -4, upper = 4, trafo = function(x) 10 ^x, requires = quote(kernel == "radial")))
# 4. MBO CONTROL
ctrl <- makeMBOControl()
ctrl <- setMBOControlTermination(ctrl, iters = 30)
tune.ctrl <- makeTuneControlMBO(mbo.control = ctrl) 

# 5. RUN OPTIMIZATION

run <- tuneParams(learner = learner, 
                  task = task, 
                  resampling = cv3, 
                  measures = acc, 
                  par.set = par.set, 
                  control = tune.ctrl)

# 6. OPTIMIZATION PATH

ys <- getOptPathY(run$opt.path)

df <- data.frame(
  iter = seq_along(ys),
  value = ys,
  best_so_far = cummax(ys)
)

# 7. PLOT PROGRESS

p1 <- ggplot(df, aes(x = iter)) +
        geom_line(aes(y = value), color = "green", linewidth = 1.2) +
        geom_point(aes(y = value), color = "black", size = 2) +
        geom_step(aes(y = best_so_far), color = "blue", linewidth = 1.4) +
        labs(
        title = "Optimization Path SVM - BO",
        x = "Iteration",
        y = "Objective Value"
        ) +
        theme_minimal(base_size = 14)

# 6. TRAIN FINAL MODEL
final_learner <- setHyperPars(learner, par.vals = run$x)
final_model <- mlr::train(final_learner, task)

preds <- predict(final_model, newdata = test)$data$response
df_preds <- data.frame(Actual = test_label, Predicted = preds)

conf_mat <- table(Predicted = df_preds$Predicted, Actual = df_preds$Actual)

# Convert to data frame for ggplot
conf_mat_df <- as.data.frame(conf_mat)

# Confusion Matrix plot
p2 <- ggplot(conf_mat_df, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = Freq), size = 8, color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  coord_fixed() +
  labs(title = "Confusion Matrix For SVM - BO (Test Set)",
       x = "Classe Predetta",
       y = "Classe Reale") +
  theme_minimal(base_size = 14) +
  theme(panel.grid = element_blank())

print(p1)
print(p2)
