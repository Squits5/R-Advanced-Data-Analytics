# R Script for Advanced Data Analysis and Visualization
library(ggplot2)
library(dplyr)
library(tidyr)
library(caret) # For machine learning tasks

# --- Data Generation (for demonstration) ---
set.seed(123)
data_points <- 500

df <- tibble(
    id = 1:data_points,
    feature1 = rnorm(data_points, mean = 50, sd = 10),
    feature2 = runif(data_points, min = 0, max = 100),
    category = sample(c("Group A", "Group B", "Group C"), data_points, replace = TRUE, prob = c(0.3, 0.4, 0.3)),
    target = 0.5 * feature1 + 0.3 * feature2 + rnorm(data_points, mean = 0, sd = 5)
)

# Introduce some outliers
df$feature1[sample(1:data_points, 5)] <- rnorm(5, mean = 150, sd = 20)

message("--- Initial Data Summary ---")
print(summary(df))

# --- Exploratory Data Analysis (EDA) ---

# Distribution of features
plot_feature_distribution <- function(data, feature_col, title) {
    ggplot(data, aes_string(x = feature_col, fill = "category")) +
        geom_density(alpha = 0.6) +
        labs(title = title, x = feature_col, y = "Density") +
        theme_minimal()
}

ggsave("feature1_distribution.png", plot_feature_distribution(df, "feature1", "Distribution of Feature 1"))
ggsave("feature2_distribution.png", plot_feature_distribution(df, "feature2", "Distribution of Feature 2"))

# Scatter plot with regression line by category
plot_scatter_by_category <- function(data, x_col, y_col, color_col, title) {
    ggplot(data, aes_string(x = x_col, y = y_col, color = color_col)) +
        geom_point(alpha = 0.6) +
        geom_smooth(method = "lm", se = FALSE) +
        labs(title = title, x = x_col, y = y_col) +
        theme_minimal()
}

ggsave("feature1_target_scatter.png", plot_scatter_by_category(df, "feature1", "target", "category", "Feature 1 vs. Target by Category"))

message("--- Correlation Matrix ---")
cor_matrix <- cor(df %>% select_if(is.numeric))
print(cor_matrix)

# --- Simple Machine Learning Model (Linear Regression) ---

# Split data
set.seed(456)
train_index <- createDataPartition(df$target, p = 0.8, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

# Train model
model_lm <- lm(target ~ feature1 + feature2 + category, data = train_data)
message("--- Linear Model Summary ---")
print(summary(model_lm))

# Make predictions
predictions <- predict(model_lm, newdata = test_data)

# Evaluate model
r_squared <- R2(predictions, test_data$target)
rmse <- RMSE(predictions, test_data$target)

message(paste0("--- Model Evaluation ---
R-squared: ", round(r_squared, 3), "
RMSE: ", round(rmse, 3)))

# Residual plot
residuals_df <- tibble(
    predicted = predictions,
    residuals = test_data$target - predictions
)

ggsave("residuals_plot.png", ggplot(residuals_df, aes(x = predicted, y = residuals)) +
    geom_point(alpha = 0.6) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    labs(title = "Residuals vs. Predicted Values", x = "Predicted", y = "Residuals") +
    theme_minimal()
)

message("Analysis complete. Plots saved as PNG files.")
