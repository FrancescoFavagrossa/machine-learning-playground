rm(list = ls()) ; graphics.off()
library(mlrMBO)
library(rgenoud)
library(dplyr)
library(ggplot2)
library(randomForest)

data <- readRDS('/Users/francescofavagrossa/Desktop/ML Project/Dataset/TicTacToe_dataset.RDS')
data$class <- as.factor(data$class)

par.set <- makeParamSet(makeIntegerParam("k", lower = 1, upper = 111))

task <- makeClassifTask(data = data, target = "class")
ctrl <- makeMBOControl()
ctrl <- setMBOControlTermination(ctrl, iters = 30)
tune.ctrl <- makeTuneControlMBO(mbo.control = ctrl)

lrn <- makeLearner("classif.knn")

# tuning
run <- tuneParams(
  learner     = lrn,
  task        = task,
  resampling  = cv3,
  measures    = acc,
  par.set     = par.set,
  control     = tune.ctrl,
  show.info   = TRUE
)

ys <- getOptPathY(run$opt.path)

df <- data.frame(
  iter = seq_along(ys),
  value = ys,
  best_so_far = cummax(ys)
)

ggplot(df, aes(x = iter)) +
  geom_line(aes(y = value), color = "green", linewidth = 1.2) +
  geom_point(aes(y = value), color = "black", size = 2) +
  geom_step(aes(y = best_so_far), color = "blue", linewidth = 1.4) +
  labs(
    title = "Optimization Path KNN - BO",
    x = "Iteration",
    y = "Objective Value"
  ) +
  theme_minimal(base_size = 14)

# Estrae i parametri testati
ks_df <- as.data.frame(getOptPathX(run$opt.path))

# Numero iterazioni
iters <- seq_len(nrow(ks_df))

# Accuracy associate
accs <- getOptPathY(run$opt.path)

# Creo dataframe per il plot
dfk <- data.frame(
  accuracy = ys,
  k = ks_df$k
)
dfk_sorted <- dfk %>% arrange(k)

ggplot(dfk_sorted, aes(x = k)) +
  geom_line(aes(y = accuracy), color = "blue", linewidth = 1.2) +
  geom_point(aes(y = accuracy), color = "black", size = 2) +
  labs(
    title = "Accuracy in funzione di K (KNN)",
    x = "Valore di K",
    y = "Accuracy"
  ) +
  theme_minimal(base_size = 14)


