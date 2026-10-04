# Fake Job Posting Detection
# Naive Bayes vs. Logistic Regression on job-posting text (bag-of-words features).
#
# Usage:
#   1. Download fake_job_postings.csv from Kaggle (link in README) into data/
#   2. Set TRAIN_FRACTION below: 0.8 for an 80/20 split, 0.5 for a 50/50 split
#   3. Run the whole script. Run it once per split, then run compare_splits.R.

TRAIN_FRACTION <- 0.8
DATA_PATH <- "data/fake_job_postings.csv"
SPLIT_LABEL <- sprintf("%d_%d", round(TRAIN_FRACTION * 100), round((1 - TRAIN_FRACTION) * 100))

# 1. Packages -----------------------------------------------------------------
required_packages <- c("caret", "tm", "SnowballC", "e1071", "naivebayes", "pROC", "ggplot2")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}
library(caret)
library(tm)
library(SnowballC)
library(e1071)   # naiveBayes()
library(pROC)
library(ggplot2)

dir.create("results", showWarnings = FALSE)

# 2. Load data ----------------------------------------------------------------
data <- read.csv(DATA_PATH, stringsAsFactors = FALSE)
str(data)
sapply(data, function(x) sum(is.na(x)))

# 3. Preprocessing ------------------------------------------------------------
# Missing text fields become empty strings
data$company_profile[is.na(data$company_profile)] <- ""
data$description[is.na(data$description)] <- ""
data$requirements[is.na(data$requirements)] <- ""
data$benefits[is.na(data$benefits)] <- ""

# Target variable. "Fake" is the class we want to detect (the "positive" class).
# Level order matters: glm() models the probability of the SECOND level,
# so with levels c("Real", "Fake") the glm output below is P(Fake).
data$fraudulent <- factor(ifelse(data$fraudulent == 1, "Fake", "Real"),
                          levels = c("Real", "Fake"))
print(table(data$fraudulent))

# 4. Text preprocessing -------------------------------------------------------
text_data <- paste(data$title, data$company_profile, data$description,
                   data$requirements, data$benefits)

corpus <- VCorpus(VectorSource(text_data))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stemDocument)
corpus <- tm_map(corpus, stripWhitespace)

# Document-Term Matrix, dropping terms that appear in fewer than ~10% of documents
dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, 0.90)

text_features <- as.data.frame(as.matrix(dtm))
text_features$fraudulent <- data$fraudulent

# 5. Train/test split ---------------------------------------------------------
set.seed(530)
trainIndex <- createDataPartition(text_features$fraudulent, p = TRAIN_FRACTION, list = FALSE)
trainData <- text_features[trainIndex, ]
testData  <- text_features[-trainIndex, ]

plot_confusion <- function(cm, title, colors, path) {
  cm_df <- as.data.frame(cm$table)
  colnames(cm_df) <- c("Predicted", "Actual", "Freq")
  p <- ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "black", size = 6) +
    scale_fill_gradientn(colours = colors) +
    labs(title = title) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  ggsave(path, p, width = 6, height = 5)
  print(p)
}

# 6a. Naive Bayes -------------------------------------------------------------
nb_model <- naiveBayes(fraudulent ~ ., data = trainData)
nb_pred  <- predict(nb_model, testData)

# positive = "Fake": precision/recall/F1 below are for detecting FAKE postings
cm_nb <- confusionMatrix(nb_pred, testData$fraudulent, positive = "Fake", mode = "everything")
print(cm_nb)
plot_confusion(cm_nb, "Confusion Matrix - Naive Bayes",
               c("#deebf7", "#9ecae1", "#3182bd", "#08519c"),
               sprintf("results/confusion_naive_bayes_%s.png", SPLIT_LABEL))

nb_precision <- unname(cm_nb$byClass["Precision"])
nb_recall    <- unname(cm_nb$byClass["Recall"])
nb_f1        <- unname(cm_nb$byClass["F1"])

nb_prob <- predict(nb_model, testData, type = "raw")[, "Fake"]
roc_nb  <- roc(testData$fraudulent, nb_prob, levels = c("Real", "Fake"), direction = "<")
auc_nb  <- as.numeric(auc(roc_nb))
png(sprintf("results/roc_naive_bayes_%s.png", SPLIT_LABEL), width = 700, height = 600)
plot(roc_nb, col = "blue", main = "ROC Curve - Naive Bayes")
dev.off()

cat("Naive Bayes (Fake = positive) Precision:", nb_precision, "\n")
cat("Naive Bayes (Fake = positive) Recall:   ", nb_recall, "\n")
cat("Naive Bayes (Fake = positive) F1:       ", nb_f1, "\n")
cat("Naive Bayes AUC:", auc_nb, "\n")

# 6b. Logistic Regression -----------------------------------------------------
glm_model <- glm(fraudulent ~ ., data = trainData, family = "binomial")
glm_probs <- predict(glm_model, testData, type = "response")      # P(Fake)
glm_pred  <- factor(ifelse(glm_probs > 0.5, "Fake", "Real"), levels = c("Real", "Fake"))

cm_glm <- confusionMatrix(glm_pred, testData$fraudulent, positive = "Fake", mode = "everything")
print(cm_glm)
plot_confusion(cm_glm, "Confusion Matrix - Logistic Regression",
               c("#e5f5e0", "#a1d99b", "#41ab5d", "#006d2c"),
               sprintf("results/confusion_logistic_regression_%s.png", SPLIT_LABEL))

glm_precision <- unname(cm_glm$byClass["Precision"])
glm_recall    <- unname(cm_glm$byClass["Recall"])
glm_f1        <- unname(cm_glm$byClass["F1"])

roc_glm <- roc(testData$fraudulent, glm_probs, levels = c("Real", "Fake"), direction = "<")
auc_glm <- as.numeric(auc(roc_glm))
png(sprintf("results/roc_logistic_regression_%s.png", SPLIT_LABEL), width = 700, height = 600)
plot(roc_glm, col = "red", main = "ROC Curve - Logistic Regression")
dev.off()

cat("Logistic Regression (Fake = positive) Precision:", glm_precision, "\n")
cat("Logistic Regression (Fake = positive) Recall:   ", glm_recall, "\n")
cat("Logistic Regression (Fake = positive) F1:       ", glm_f1, "\n")
cat("Logistic Regression AUC:", auc_glm, "\n")

# 7. 5-fold cross-validation (ROC-AUC) ----------------------------------------
train_control_5fold <- trainControl(method = "cv", number = 5,
                                    classProbs = TRUE,
                                    summaryFunction = twoClassSummary)

glm_cv_5fold <- train(fraudulent ~ ., data = text_features,
                      method = "glm", family = "binomial",
                      trControl = train_control_5fold, metric = "ROC")
print(glm_cv_5fold)

nb_cv_5fold <- train(fraudulent ~ ., data = text_features,
                     method = "naive_bayes",
                     trControl = train_control_5fold, metric = "ROC")
print(nb_cv_5fold)

cv_roc_glm <- max(glm_cv_5fold$results$ROC)
cv_roc_nb  <- max(nb_cv_5fold$results$ROC)   # best across caret's tuning grid
cat("Logistic Regression mean ROC-AUC (5-fold):", cv_roc_glm, "\n")
cat("Naive Bayes mean ROC-AUC (5-fold, best tune):", cv_roc_nb, "\n")

# 8. Save results -------------------------------------------------------------
split_name <- sprintf("%d/%d", round(TRAIN_FRACTION * 100), round((1 - TRAIN_FRACTION) * 100))
results <- data.frame(
  Model     = c("Naive Bayes", "Logistic Regression"),
  Split     = split_name,
  Precision = c(nb_precision, glm_precision),
  Recall    = c(nb_recall, glm_recall),
  F1        = c(nb_f1, glm_f1),
  AUC       = c(auc_nb, auc_glm),
  CV_AUC    = c(cv_roc_nb, cv_roc_glm)
)
print(results)
write.csv(results, sprintf("results/metrics_%s.csv", SPLIT_LABEL), row.names = FALSE)
