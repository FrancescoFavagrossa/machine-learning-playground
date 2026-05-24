rm(list = ls()) ; graphics.off()
library(GauPro)
library(ggplot2)
library(gridExtra)
library(caret)

# 1. Caricamento dati e preparazione
data <- readRDS("/Users/francescofavagrossa/Desktop/ML Project/Dataset/TicTacToe_dataset.RDS")
data$class <- as.factor(data$class)

# 2. Split train/test
set.seed(123)  # Per riproducibilità
train_idx <- createDataPartition(data$class, p=0.9, list=FALSE)

# Dataset di training completo
ds <- data[train_idx, ]

# Dataset di test (features e label separati)
test <- data[-train_idx, -ncol(data)]
test_label <- data[-train_idx, ]$class

# 3. Preparazione per Gaussian Process
# Estrai features (X) e target (y)
X_train <- as.matrix(ds[, -ncol(ds)])  # Tutte le colonne tranne l'ultima (class)
y_train_factor <- ds$class

# Converti y in numerico (mantiene 1 e -1)
y_train <- as.numeric(as.character(y_train_factor))  # 1 e -1

# Test set
X_test <- as.matrix(test)
y_test <- as.numeric(as.character(test_label))  # 1 e -1

# 4. Fit Gaussian Process


kern <- Gaussian$new(D = ncol(X_train))  # Kernel Gaussiano
gp_model <- GauPro_kernel_model$new(
  X = X_train,
  Z = y_train,
  kernel = kern,
  verbose = 1
)

# 5. Predizioni
test_preds <- gp_model$predict(X_test)
test_pred_class <- ifelse(test_preds > 0, 1, -1)

# Emp. Error (training)
train_preds <- gp_model$predict(X_train)
train_pred_class <- ifelse(train_preds > 0, 1, -1)

# 6. Accuracy
train_accuracy <- mean(train_pred_class == y_train)
test_accuracy <- mean(test_pred_class == y_test)

cat("RISULTATI\n")
cat("Training Accuracy:", round(train_accuracy, 4), "\n")
cat("Test Accuracy:", round(test_accuracy, 4), "\n")


# Grafico: Accuracy Train vs Test
accuracy_df <- data.frame(
  Set = c("Training", "Test"),
  Accuracy = c(train_accuracy, test_accuracy)
)
p1 <- ggplot(accuracy_df, aes(x = Set, y = Accuracy, fill = Set)) +
  geom_bar(stat = "identity", alpha = 0.7) +
  geom_text(aes(label = round(Accuracy, 4)), vjust = -0.5, size = 5) +
  scale_fill_manual(values = c("Training" = "steelblue", "Test" = "darkgreen")) +
  labs(title = "Accuracy: Training vs Test", x = "", y = "Accuracy") +
  theme_minimal() +
  ylim(0, 1) +
  theme(legend.position = "none")

# Grafico: Confusion Matrix (Test Set)
conf_matrix <- t(table(Predicted = y_test, Actual = test_pred_class))
conf_df <- as.data.frame(conf_matrix)

p2 <- ggplot(conf_df, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = Freq), size = 8, color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  coord_fixed() +
  labs(title = "Confusion Matrix (Test Set)",
       x = "Classe Predetta",
       y = "Classe Reale") +
  theme_minimal(base_size = 14) +
  theme(panel.grid = element_blank())

grid.arrange(p1,p2)
cat("\nLearned Kernel Hyperparameters:\n")
print(gp_model$kernel$beta)

# 2. Get the mean function values for any point
# Example: predict on a grid for specific features
mean_predictions <- gp_model$predict(X_test, se.fit = TRUE)
means <- mean_predictions$mean
std_errors <- mean_predictions$se

# 3. Visualize uncertainty for test predictions
uncertainty_df <- data.frame(
  Index = 1:length(means),
  Mean = means,
  Lower = means - 2*std_errors,
  Upper = means + 2*std_errors,
  True = y_test
)

ggplot(uncertainty_df, aes(x = Index)) +
  geom_ribbon(aes(ymin = Lower, ymax = Upper), alpha = 0.3, fill = "blue") +
  geom_line(aes(y = Mean), color = "blue", size = 1) +
  geom_point(aes(y = True), color = "red", alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "GP Predictions with Uncertainty",
       y = "Predicted Value",
       x = "Test Sample Index") +
  theme_minimal()

gp_model$kernel$beta
# 1. Estrai gli iperparametri del kernel
kernel_hyperparams <- gp_model$kernel$beta

# 2. Determina il numero di features (D = 9 in questo caso)
D_features <- ncol(X_train)

# 3. Estrai i length scale parameters (i primi D elementi)
length_scales <- kernel_hyperparams[1:D_features]

# 4. Crea un dataframe per ggplot
# Assegna nomi alle feature (qui usiamo nomi generici da F1 a F9)
feature_names <- colnames(X_train)
if (is.null(feature_names)) {
  feature_names <- paste0("F", 1:D_features)
}

ls_df <- data.frame(
  Feature = feature_names,
  LengthScale = length_scales
)

# 5. Genera il grafico
p_ls <- ggplot(ls_df, aes(x = reorder(Feature, LengthScale), y = LengthScale, fill = LengthScale)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  geom_text(aes(label = round(LengthScale, 3)), vjust = -0.5, size = 4) +
  labs(title = "Length Scale Parameters appresi per Feature",
       subtitle = "Il valore del Length Scale indica l'importanza relativa della feature",
       x = "Feature (Posizioni della Griglia)",
       y = "Length Scale (l)") +
  scale_fill_gradient(low = "yellowgreen", high = "darkolivegreen") +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(face = "bold"))

# Stampa il grafico
print(p_ls)

