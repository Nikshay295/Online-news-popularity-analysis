
# Load necessary libraries
library(readr)
library(dplyr)
library(caret)
library(MASS)
library(pscl)

# Load dataset
data <- read.csv("C:\\Users\\Nikshay\\Desktop\\OnlineNewsPopularity.csv")

# Select only the required columns
selected_data <- data %>% dplyr::select(
  n_tokens_title, n_tokens_content, num_hrefs, num_self_hrefs, num_imgs, 
  num_videos, average_token_length, num_keywords, data_channel_is_lifestyle, 
  data_channel_is_entertainment, data_channel_is_bus, shares
)

# Convert categorical variables to factors
selected_data <- selected_data %>%
  mutate(
    data_channel_is_lifestyle = as.factor(data_channel_is_lifestyle),
    data_channel_is_entertainment = as.factor(data_channel_is_entertainment),
    data_channel_is_bus = as.factor(data_channel_is_bus)
  )

# Remove rows with missing or invalid values
cleaned_data <- na.omit(selected_data)

# Split the data into 70% training and 30% testing sets
set.seed(123)
train_index <- createDataPartition(cleaned_data$shares, p = 0.7, list = FALSE)
train_data <- cleaned_data[train_index, ]
test_data <- cleaned_data[-train_index, ]

# Function to calculate performance metrics
calculate_metrics <- function(model, data, actual) {
  predictions <- predict(model, newdata = data, type = "response")
  residuals <- actual - predictions
  mse <- mean(residuals^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(residuals))
  aic <- AIC(model)
  r_squared <- 1 - sum(residuals^2) / sum((actual - mean(actual))^2)
  list(MSE = mse, RMSE = rmse, MAE = mae, AIC = aic, R_squared = r_squared)
}

# 1. Linear Regression
linear_model <- lm(shares ~ n_tokens_title + n_tokens_content + num_hrefs +
                     num_self_hrefs + num_imgs + num_videos + average_token_length +
                     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment +
                     data_channel_is_bus, data = train_data)
summary(linear_model)

# Diagnostic plots for Linear Regression
par(mfrow = c(2, 2))
plot(linear_model, main = "Linear Regression Diagnostics")

# Evaluate Linear Regression
linear_metrics <- calculate_metrics(linear_model, test_data, test_data$shares)

# 2. Poisson Regression
poisson_model <- glm(shares ~ n_tokens_title + n_tokens_content + num_hrefs +
                       num_self_hrefs + num_imgs + num_videos + average_token_length +
                       num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment +
                       data_channel_is_bus, family = poisson(link = "log"), data = train_data)
summary(poisson_model)

# Diagnostic plots for Poisson Regression
par(mfrow = c(2, 2))
plot(predict(poisson_model, type = "response"), residuals(poisson_model, type = "deviance"),
     main = "Poisson Residuals vs Fitted", xlab = "Fitted Values", ylab = "Residuals")
plot(poisson_model, which = 2, main = "Poisson Normal Q-Q Plot")
plot(poisson_model, which = 3, main = "Poisson Scale-Location Plot")
plot(poisson_model, which = 4, main = "Poisson Residuals vs Leverage")

# Evaluate Poisson Regression
poisson_metrics <- calculate_metrics(poisson_model, test_data, test_data$shares)

# 3. Quasi-Poisson Regression
quasi_model <- glm(shares ~ n_tokens_title + n_tokens_content + num_hrefs +
                     num_self_hrefs + num_imgs + num_videos + average_token_length +
                     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment +
                     data_channel_is_bus, family = quasipoisson(link = "log"), data = train_data)
summary(quasi_model)

# Diagnostic plots for Quasi-Poisson Regression
par(mfrow = c(2, 2))
plot(predict(quasi_model, type = "response"), residuals(quasi_model, type = "deviance"),
     main = "Quasi-Poisson Residuals vs Fitted", xlab = "Fitted Values", ylab = "Residuals")
plot(quasi_model, which = 2, main = "Quasi-Poisson Normal Q-Q Plot")
plot(quasi_model, which = 3, main = "Quasi-Poisson Scale-Location Plot")
plot(quasi_model, which = 4, main = "Quasi-Poisson Residuals vs Leverage")

# Evaluate Quasi-Poisson Regression
quasi_metrics <- calculate_metrics(quasi_model, test_data, test_data$shares)

# 4. Negative Binomial Regression
nb_model <- glm.nb(shares ~ n_tokens_title + n_tokens_content + num_hrefs +
                     num_self_hrefs + num_imgs + num_videos + average_token_length +
                     num_keywords + data_channel_is_lifestyle + data_channel_is_entertainment +
                     data_channel_is_bus, data = train_data)
summary(nb_model)

# Diagnostic plots for Negative Binomial Regression
par(mfrow = c(2, 2))
plot(predict(nb_model, type = "response"), residuals(nb_model, type = "deviance"),
     main = "Negative Binomial Residuals vs Fitted", xlab = "Fitted Values", ylab = "Residuals")
plot(nb_model, which = 2, main = "Negative Binomial Normal Q-Q Plot")
plot(nb_model, which = 3, main = "Negative Binomial Scale-Location Plot")
plot(nb_model, which = 4, main = "Negative Binomial Residuals vs Leverage")

# Evaluate Negative Binomial Regression
nb_metrics <- calculate_metrics(nb_model, test_data, test_data$shares)

# Combine Metrics
all_metrics <- data.frame(
  Model = c("Linear", "Poisson", "Quasi-Poisson", "Negative Binomial"),
  MSE = c(linear_metrics$MSE, poisson_metrics$MSE, quasi_metrics$MSE, nb_metrics$MSE),
  RMSE = c(linear_metrics$RMSE, poisson_metrics$RMSE, quasi_metrics$RMSE, nb_metrics$RMSE),
  MAE = c(linear_metrics$MAE, poisson_metrics$MAE, quasi_metrics$MAE, nb_metrics$MAE),
  AIC = c(linear_metrics$AIC, poisson_metrics$AIC, quasi_metrics$AIC, nb_metrics$AIC),
  R_Squared = c(linear_metrics$R_squared, poisson_metrics$R_squared, quasi_metrics$R_squared, nb_metrics$R_squared)
)

print("Model Comparison:")
print(all_metrics)

# Determine Best Model
best_model <- all_metrics %>%
  filter(AIC == min(AIC, na.rm = TRUE)) %>%
  pull(Model)
print(paste("The best model is:", best_model))
