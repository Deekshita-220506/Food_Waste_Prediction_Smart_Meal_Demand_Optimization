# ============================================================
# FOOD WASTE PREDICTION AND SMART MEAL DEMAND OPTIMIZATION
# Student: Gondela Durga Gnana Deekshita
# Roll No.: 160124771009
# ============================================================
#
# Purpose:
#   1. Import and inspect meal-demand / food-waste data
#   2. Clean and preprocess the data
#   3. Perform exploratory data analysis and visualization
#   4. Predict meal demand using Random Forest Regression
#   5. Evaluate prediction performance
#   6. Convert predicted demand into a smart meal-preparation target
#
# Expected CSV file:
#   food_waste_meal_demand.csv
#
# Recommended columns:
#   Date, Meal_Type, Day_of_Week, Meals_Prepared, Meals_Served,
#   Leftover_Quantity, Consumers, Event_or_Holiday
#
# If the CSV is not found, a small demonstration dataset is generated
# automatically so that the script can still be tested.
# ============================================================


# -----------------------------
# 1. INSTALL / LOAD PACKAGES
# -----------------------------

required_packages <- c(
  "readr",
  "dplyr",
  "ggplot2",
  "caret",
  "randomForest"
)

new_packages <- required_packages[
  !(required_packages %in% installed.packages()[, "Package"])
]

if (length(new_packages) > 0) {
  install.packages(new_packages, repos = "https://cloud.r-project.org")
}

library(readr)
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)


# -----------------------------
# 2. IMPORT DATA
# -----------------------------

file_name <- "food_waste_meal_demand.csv"

if (file.exists(file_name)) {

  food <- read_csv(file_name, show_col_types = FALSE)

} else {

  # Demonstration data for testing the script.
  # Replace this with the actual dataset before final submission.

  set.seed(123)

  n <- 300

  food <- data.frame(
    Date = seq.Date(
      from = as.Date("2026-01-01"),
      by = "day",
      length.out = n
    ),
    Meal_Type = sample(
      c("Breakfast", "Lunch", "Dinner"),
      n,
      replace = TRUE
    ),
    Consumers = sample(
      80:250,
      n,
      replace = TRUE
    ),
    Event_or_Holiday = sample(
      c("No", "Yes"),
      n,
      replace = TRUE,
      prob = c(0.9, 0.1)
    )
  )

  food$Day_of_Week <- weekdays(food$Date)

  # Generate realistic demonstration meal demand.
  meal_effect <- ifelse(food$Meal_Type == "Lunch", 25,
                        ifelse(food$Meal_Type == "Dinner", 10, -5))

  event_effect <- ifelse(food$Event_or_Holiday == "Yes", -15, 0)

  food$Meals_Served <- pmax(
    1,
    round(food$Consumers + meal_effect + event_effect +
            rnorm(n, mean = 0, sd = 12))
  )

  food$Meals_Prepared <- pmax(
    food$Meals_Served,
    food$Meals_Served + sample(0:35, n, replace = TRUE)
  )

  food$Leftover_Quantity <- pmax(
    0,
    food$Meals_Prepared - food$Meals_Served
  )

  write_csv(food, "food_waste_demo_data.csv")

  message(
    "CSV file was not found. Demonstration data was generated and saved as food_waste_demo_data.csv."
  )
}


# -----------------------------
# 3. INITIAL DATA INSPECTION
# -----------------------------

cat("\n================ DATASET INFORMATION ================\n")

cat("Rows:", nrow(food), "\n")
cat("Columns:", ncol(food), "\n")

cat("\nColumn names:\n")
print(names(food))

cat("\nStructure:\n")
str(food)

cat("\nSummary:\n")
print(summary(food))


# -----------------------------
# 4. DATA PREPARATION
# -----------------------------

# Convert date if available.
if ("Date" %in% names(food)) {
  food$Date <- as.Date(food$Date)
}

# Create Day_of_Week if it is not already present.
if ("Day_of_Week" %in% names(food)) {
  food$Day_of_Week <- as.factor(food$Day_of_Week)
} else if ("Date" %in% names(food)) {
  food$Day_of_Week <- as.factor(weekdays(food$Date))
}

# Convert categorical variables.
if ("Meal_Type" %in% names(food)) {
  food$Meal_Type <- as.factor(food$Meal_Type)
}

if ("Event_or_Holiday" %in% names(food)) {
  food$Event_or_Holiday <- as.factor(food$Event_or_Holiday)
}


# -----------------------------
# 5. MISSING VALUES
# -----------------------------

cat("\n================ MISSING VALUES ================\n")

missing_values <- colSums(is.na(food))
print(missing_values)

# Remove rows with missing values for the variables used in modelling.
model_columns <- intersect(
  c(
    "Meals_Served",
    "Meal_Type",
    "Day_of_Week",
    "Consumers",
    "Event_or_Holiday"
  ),
  names(food)
)

food_model <- food %>%
  select(all_of(model_columns)) %>%
  na.omit()


# -----------------------------
# 6. DUPLICATE RECORDS
# -----------------------------

cat("\n================ DUPLICATES ================\n")

duplicate_count <- sum(duplicated(food))

cat("Duplicate rows:", duplicate_count, "\n")

food <- food[!duplicated(food), ]


# -----------------------------
# 7. CHECK INCORRECT / NEGATIVE VALUES
# -----------------------------

cat("\n================ VALUE VALIDATION ================\n")

numeric_columns <- intersect(
  c(
    "Meals_Prepared",
    "Meals_Served",
    "Leftover_Quantity",
    "Consumers"
  ),
  names(food)
)

for (col in numeric_columns) {
  negative_count <- sum(food[[col]] < 0, na.rm = TRUE)
  cat(col, "- negative values:", negative_count, "\n")
}


# -----------------------------
# 8. FEATURE ENGINEERING
# -----------------------------

if (all(
  c("Meals_Prepared", "Leftover_Quantity") %in% names(food)
)) {

  food$Waste_Rate <- ifelse(
    food$Meals_Prepared > 0,
    food$Leftover_Quantity / food$Meals_Prepared,
    0
  )

  food$Waste_Rate <- pmin(
    pmax(food$Waste_Rate, 0),
    1
  )
}

if (all(
  c("Meals_Prepared", "Meals_Served") %in% names(food)
)) {

  food$Service_Rate <- ifelse(
    food$Meals_Prepared > 0,
    food$Meals_Served / food$Meals_Prepared,
    0
  )

  food$Service_Rate <- pmin(
    pmax(food$Service_Rate, 0),
    1
  )
}


# -----------------------------
# 9. DESCRIPTIVE STATISTICS
# -----------------------------

cat("\n================ DESCRIPTIVE STATISTICS ================\n")

summary_table <- data.frame(
  Measure = character(),
  Value = numeric(),
  stringsAsFactors = FALSE
)

if ("Meals_Prepared" %in% names(food)) {
  summary_table <- rbind(
    summary_table,
    data.frame(
      Measure = "Average Meals Prepared",
      Value = mean(food$Meals_Prepared, na.rm = TRUE)
    )
  )
}

if ("Meals_Served" %in% names(food)) {
  summary_table <- rbind(
    summary_table,
    data.frame(
      Measure = "Average Meals Served",
      Value = mean(food$Meals_Served, na.rm = TRUE)
    )
  )
}

if ("Leftover_Quantity" %in% names(food)) {
  summary_table <- rbind(
    summary_table,
    data.frame(
      Measure = "Average Leftover Quantity",
      Value = mean(food$Leftover_Quantity, na.rm = TRUE)
    )
  )
}

if ("Waste_Rate" %in% names(food)) {
  summary_table <- rbind(
    summary_table,
    data.frame(
      Measure = "Average Waste Rate",
      Value = mean(food$Waste_Rate, na.rm = TRUE)
    )
  )
}

print(summary_table)


# -----------------------------
# 10. ANALYSIS BY MEAL TYPE
# -----------------------------

if ("Meal_Type" %in% names(food)) {

  meal_summary <- food %>%
    group_by(Meal_Type) %>%
    summarise(
      Average_Prepared = if ("Meals_Prepared" %in% names(food))
        mean(Meals_Prepared, na.rm = TRUE) else NA_real_,

      Average_Served = if ("Meals_Served" %in% names(food))
        mean(Meals_Served, na.rm = TRUE) else NA_real_,

      Average_Leftover = if ("Leftover_Quantity" %in% names(food))
        mean(Leftover_Quantity, na.rm = TRUE) else NA_real_,

      Average_Waste_Rate = if ("Waste_Rate" %in% names(food))
        mean(Waste_Rate, na.rm = TRUE) else NA_real_,

      .groups = "drop"
    )

  cat("\n================ MEAL TYPE ANALYSIS ================\n")
  print(meal_summary)
}


# -----------------------------
# 11. VISUALIZATION 1:
#     AVERAGE DEMAND BY MEAL TYPE
# -----------------------------

if (all(c("Meal_Type", "Meals_Served") %in% names(food))) {

  p1 <- ggplot(
    food,
    aes(
      x = Meal_Type,
      y = Meals_Served
    )
  ) +
    stat_summary(
      fun = mean,
      geom = "bar"
    ) +
    labs(
      title = "Average Meal Demand by Meal Type",
      x = "Meal Type",
      y = "Average Meals Served"
    ) +
    theme_minimal()

  print(p1)

  ggsave(
    "01_average_meal_demand.png",
    p1,
    width = 8,
    height = 5
  )
}


# -----------------------------
# 12. VISUALIZATION 2:
#     LEFTOVER BY MEAL TYPE
# -----------------------------

if (all(
  c("Meal_Type", "Leftover_Quantity") %in% names(food)
)) {

  p2 <- ggplot(
    food,
    aes(
      x = Meal_Type,
      y = Leftover_Quantity
    )
  ) +
    stat_summary(
      fun = mean,
      geom = "bar"
    ) +
    labs(
      title = "Average Leftover Quantity by Meal Type",
      x = "Meal Type",
      y = "Average Leftover Quantity"
    ) +
    theme_minimal()

  print(p2)

  ggsave(
    "02_average_leftover_by_meal.png",
    p2,
    width = 8,
    height = 5
  )
}


# -----------------------------
# 13. VISUALIZATION 3:
#     DEMAND OVER TIME
# -----------------------------

if (all(
  c("Date", "Meals_Served") %in% names(food)
)) {

  p3 <- ggplot(
    food,
    aes(
      x = Date,
      y = Meals_Served
    )
  ) +
    geom_line() +
    labs(
      title = "Meal Demand Over Time",
      x = "Date",
      y = "Meals Served"
    ) +
    theme_minimal()

  print(p3)

  ggsave(
    "03_meal_demand_over_time.png",
    p3,
    width = 9,
    height = 5
  )
}


# -----------------------------
# 14. VISUALIZATION 4:
#     DEMAND BY DAY
# -----------------------------

if (all(
  c("Day_of_Week", "Meals_Served") %in% names(food)
)) {

  p4 <- ggplot(
    food,
    aes(
      x = Day_of_Week,
      y = Meals_Served
    )
  ) +
    geom_boxplot() +
    labs(
      title = "Meal Demand Distribution by Day of Week",
      x = "Day of Week",
      y = "Meals Served"
    ) +
    theme_minimal()

  print(p4)

  ggsave(
    "04_demand_by_day.png",
    p4,
    width = 9,
    height = 5
  )
}


# -----------------------------
# 15. VISUALIZATION 5:
#     PREPARED VS LEFTOVER
# -----------------------------

if (all(
  c("Meals_Prepared", "Leftover_Quantity") %in% names(food)
)) {

  p5 <- ggplot(
    food,
    aes(
      x = Meals_Prepared,
      y = Leftover_Quantity
    )
  ) +
    geom_point(alpha = 0.6) +
    geom_smooth(
      method = "lm",
      se = FALSE
    ) +
    labs(
      title = "Meals Prepared vs Leftover Quantity",
      x = "Meals Prepared",
      y = "Leftover Quantity"
    ) +
    theme_minimal()

  print(p5)

  ggsave(
    "05_prepared_vs_leftover.png",
    p5,
    width = 8,
    height = 5
  )
}


# -----------------------------
# 16. PREPARE MODEL DATA
# -----------------------------

required_model_columns <- c(
  "Meals_Served",
  "Meal_Type",
  "Day_of_Week",
  "Consumers"
)

if (all(required_model_columns %in% names(food))) {

  model_data <- food %>%
    select(
      Meals_Served,
      Meal_Type,
      Day_of_Week,
      Consumers,
      any_of("Event_or_Holiday")
    ) %>%
    na.omit()

} else {

  stop(
    paste(
      "The dataset must contain at least:",
      paste(required_model_columns, collapse = ", ")
    )
  )
}


# -----------------------------
# 17. TRAIN / TEST SPLIT
# -----------------------------

set.seed(123)

train_index <- createDataPartition(
  model_data$Meals_Served,
  p = 0.80,
  list = FALSE
)

train <- model_data[train_index, ]
test <- model_data[-train_index, ]

cat("\n================ DATA SPLIT ================\n")
cat("Training rows:", nrow(train), "\n")
cat("Testing rows:", nrow(test), "\n")


# -----------------------------
# 18. RANDOM FOREST MODEL
# -----------------------------

set.seed(123)

rf_model <- randomForest(
  Meals_Served ~ .,
  data = train,
  ntree = 300,
  importance = TRUE
)

cat("\n================ RANDOM FOREST MODEL ================\n")
print(rf_model)


# -----------------------------
# 19. PREDICTIONS
# -----------------------------

predicted_demand <- predict(
  rf_model,
  newdata = test
)

results <- data.frame(
  Actual = test$Meals_Served,
  Predicted = predicted_demand
)

print(head(results))


# -----------------------------
# 20. MODEL EVALUATION
# -----------------------------

rmse <- sqrt(
  mean(
    (results$Actual - results$Predicted)^2
  )
)

mae <- mean(
  abs(
    results$Actual - results$Predicted
  )
)

r_squared <- cor(
  results$Actual,
  results$Predicted
)^2

evaluation <- data.frame(
  Metric = c(
    "RMSE",
    "MAE",
    "R-Squared"
  ),
  Value = c(
    rmse,
    mae,
    r_squared
  )
)

cat("\n================ MODEL EVALUATION ================\n")
print(evaluation)

write_csv(
  evaluation,
  "model_evaluation_results.csv"
)


# -----------------------------
# 21. ACTUAL VS PREDICTED PLOT
# -----------------------------

p6 <- ggplot(
  results,
  aes(
    x = Actual,
    y = Predicted
  )
) +
  geom_point(alpha = 0.7) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed"
  ) +
  labs(
    title = "Actual vs Predicted Meal Demand",
    x = "Actual Meals Served",
    y = "Predicted Meals Served"
  ) +
  theme_minimal()

print(p6)

ggsave(
  "06_actual_vs_predicted.png",
  p6,
  width = 8,
  height = 5
)


# -----------------------------
# 22. FEATURE IMPORTANCE
# -----------------------------

importance_values <- importance(rf_model)

importance_df <- data.frame(
  Variable = rownames(importance_values),
  Importance = importance_values[, 1]
)

importance_df <- importance_df %>%
  arrange(desc(Importance))

cat("\n================ FEATURE IMPORTANCE ================\n")
print(importance_df)

p7 <- ggplot(
  importance_df,
  aes(
    x = reorder(Variable, Importance),
    y = Importance
  )
) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Random Forest Feature Importance",
    x = "Variable",
    y = "Importance"
  ) +
  theme_minimal()

print(p7)

ggsave(
  "07_feature_importance.png",
  p7,
  width = 8,
  height = 5
)


# -----------------------------
# 23. SMART MEAL DEMAND OPTIMIZATION
# -----------------------------

# A small safety factor is used to reduce the risk of
# under-preparation. This value should be validated using
# actual operational results.

safety_factor <- 0.05

optimization_results <- data.frame(
  Actual_Demand = test$Meals_Served,
  Predicted_Demand = predicted_demand
)

optimization_results$Recommended_Preparation <- ceiling(
  optimization_results$Predicted_Demand *
    (1 + safety_factor)
)

optimization_results$Estimated_Surplus <- pmax(
  optimization_results$Recommended_Preparation -
    optimization_results$Actual_Demand,
  0
)

optimization_results$Estimated_Shortage <- pmax(
  optimization_results$Actual_Demand -
    optimization_results$Recommended_Preparation,
  0
)

cat("\n================ SMART MEAL OPTIMIZATION ================\n")
print(head(optimization_results, 10))

write_csv(
  optimization_results,
  "smart_meal_optimization_results.csv"
)


# -----------------------------
# 24. SUMMARY OF RESULTS
# -----------------------------

cat("\n========================================================\n")
cat("FINAL CASE STUDY SUMMARY\n")
cat("========================================================\n")

cat("Student: Gondela Durga Gnana Deekshita\n")
cat("Roll No.: 160124771009\n")
cat("Dataset rows:", nrow(food), "\n")
cat("Dataset columns:", ncol(food), "\n")
cat("RMSE:", round(rmse, 3), "\n")
cat("MAE:", round(mae, 3), "\n")
cat("R-Squared:", round(r_squared, 3), "\n")
cat("Safety factor:", safety_factor * 100, "%\n")

cat("\nGenerated output files:\n")
cat("- model_evaluation_results.csv\n")
cat("- smart_meal_optimization_results.csv\n")
cat("- 01_average_meal_demand.png\n")
cat("- 02_average_leftover_by_meal.png\n")
cat("- 03_meal_demand_over_time.png\n")
cat("- 04_demand_by_day.png\n")
cat("- 05_prepared_vs_leftover.png\n")
cat("- 06_actual_vs_predicted.png\n")
cat("- 07_feature_importance.png\n")

cat("\nIMPORTANT:\n")
cat("Replace the demonstration data with your actual dataset before final submission.\n")
cat("Update the documented results in the Word/R Markdown report using the actual R output.\n")
