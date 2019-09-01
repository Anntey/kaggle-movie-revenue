# Kaggle: TMDB Box Office Prediction ([link](https://www.kaggle.com/c/tmdb-box-office-prediction))

Data: 7 398 movies with a variety of metadata

Task: predict movie's overall box office revenue

Evaluation: Root-Mean-Squared-Logarithmic-Error (RMSLE)

Solution: model averaging over (1) XGBoost and (2) CatBoostRegressor

Success: 2.03 RMSLE

![](shap_summary.png)
