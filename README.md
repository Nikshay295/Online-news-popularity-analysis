# Online News Popularity Analysis

This project aims to predict the number of shares a news article receives on social media using the **Online News Popularity** dataset. The analysis focuses on handling **count data** using classical statistical models and addressing **overdispersion**.

## Models Applied

- Linear Regression (baseline)
- Poisson Regression
- Quasi-Poisson Regression
- Negative Binomial Regression

## Dataset

- Source: UCI Machine Learning Repository
- Size: ~40K rows
- Target: `shares` (number of shares for a news article)

## Key Insights

- Linear Regression performed poorly due to the non-normal, overdispersed nature of count data.
- Poisson regression showed extreme overdispersion.
- Quasi-Poisson helped but lacked AIC comparison.
- **Negative Binomial** regression (Theta ≈ 0.92) gave the best fit in terms of AIC, RMSE, and interpretability.

## Files Included

- `online_news_model.R`: R script with model fitting, evaluation, and plots.
- `categorical analysis project.docx`: Detailed project report.
- `figures/`: Optional plots and diagnostics (if uploaded)

## Future Improvements

- Try Zero-Inflated Models (ZIP/ZINB) for better fit
- Use ML models (Random Forest, GBM)
- Add interaction terms or time features

---

##  Project Timeline

- **Duration:** 2 weeks
- **Tools:** R, caret, MASS, pscl
- **Contributor:** Nikshay Policepatel
