# Compare model performance across the 80/20 and 50/50 splits.
# Run fake_job_detection.R once with TRAIN_FRACTION <- 0.8 and once with 0.5 first.

for (pkg in c("ggplot2", "reshape2")) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}
library(ggplot2)
library(reshape2)

res <- rbind(read.csv("results/metrics_80_20.csv"),
             read.csv("results/metrics_50_50.csv"))

long <- melt(res[, c("Model", "Split", "Precision", "Recall", "F1", "AUC")],
             id.vars = c("Model", "Split"),
             variable.name = "Metric", value.name = "Value")

p <- ggplot(long, aes(x = Metric, y = Value, fill = Split)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ Model) +
  ylim(0, 1) +
  labs(title = "Fake-posting detection: 80/20 vs 50/50 split",
       subtitle = "Precision, recall and F1 are for the Fake class",
       y = "Score", x = "") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set2")

ggsave("results/split_comparison.png", p, width = 9, height = 5)
print(p)
