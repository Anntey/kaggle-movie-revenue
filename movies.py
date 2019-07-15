
import ast
import shap
import numpy as np
import pandas as pd
import seaborn as sns 
import xgboost as xgb
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

##################
## Loading data ##
##################

train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

train_extra_feats = pd.read_csv("./input/TrainAdditionalFeatures.csv") # additional features
test_extra_feats = pd.read_csv("./input/TestAdditionalFeatures.csv")

#########################
## Feature engineering ##
#########################

train[["release_month", "release_day", "release_year"]] = train["release_date"].str.split("/", expand = True).astype(int) # split date
train.loc[train["release_year"] <= 19, "release_year"] += 2000 # released 2000-2019
train.loc[train["release_year"] < 2000, "release_year"] += 1900 # released 1920-1999

release_date = pd.to_datetime(train["release_date"]) 
train["release_dayofweek"] = release_date.dt.dayofweek # extract day of week
train["release_quarter"] = release_date.dt.quarter # extract quarter

train = pd.merge(train, train_extra_feats, how = "left", on = "imdb_id") # now ready to join extra features

dict_cols_list = [
        "genres",
        "production_companies",
        "production_countries",
        "spoken_languages",
        "cast",
        "crew",
        "Keywords"
]

for column in dict_cols_list: # converts json columns into python dictionaries
    train[column] = train[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x))
    test[column] = test[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x))

train["has_homepage"] = 0
train.loc[train["homepage"].isnull() == False, "has_homepage"] = 1 # dummify homepage property

train["language_en"] = 0
train.loc[train["original_language"] == "en", "language_en"] = 1 # dummify language property

train["has_collection"] = 0
train.loc[train["belongs_to_collection"].isnull() == False, "has_collection"] = 1 # dummify collection property

train["has_tagline"] = 0
train.loc[train["tagline"].isnull() == False, "has_tagline"] = 1 # dummify homepage property

train["num_countries"] = train["production_countries"].apply(lambda x: len(x) if x != {} else 0) # some crude features
train["num_genres"] = train["genres"].apply(lambda x: len(x) if x != {} else 0)
train["num_cast"] = train["cast"].apply(lambda x: len(x) if x != {} else 0)
train["num_crew"] = test["crew"].apply(lambda x: len(x) if x != {} else 0)
train["num_keywords"] = train["Keywords"].apply(lambda x: len(x) if x != {} else 0)
train["num_companies"] = train["production_companies"].apply(lambda x: len(x) if x != {} else 0)
train["num_languages"] = train["spoken_languages"].apply(lambda x: len(x) if x != {} else 0)

sns.distplot(train["revenue"])
train["revenue"] = np.log1p(train["revenue"]) # fix skewness

train["budget"] = train["budget"] + train["budget"] * 0.018 * (2018 - train["release_year"]) # adjusting for inflation
sns.distplot(train["budget"])
train["budget"] = np.log1p(train["budget"]) # fix skewness

train["status"].value_counts()
not_released = train[train["status"] != "Released"].index
train = train.drop(not_released) # remove movies not released yet

train = train[[ # select features to be used
        "budget",
        "popularity",
        "popularity2",
        "runtime",
        "rating",
        "totalVotes",
        "release_month",
        "release_day",
        "release_dayofweek",
        "release_year",
        "release_quarter",
        "has_homepage",
        "has_collection",
        "has_tagline",
        "language_en",
        "num_countries",
        "num_cast",
        "num_genres",
        "num_crew",
        "num_companies",
        "num_languages",
        "num_keywords",
        "revenue",
        ]
]
    
train.isna().sum()    
    
train = train.dropna(axis = 0) # drop rows with missing values

x = train.drop(["revenue"], axis = 1)
y = train["revenue"]

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2)

######################
## Visualizing data ##
######################

sns.scatterplot(x = "budget", y = "revenue", data = train, alpha = 0.4) 
sns.scatterplot(x = "popularity", y = "revenue", data = train, alpha = 0.4)
sns.scatterplot(x = "runtime", y = "revenue", data = train, alpha = 0.4)

sns.countplot(x = "release_year", data = train)
sns.countplot(x = "release_month", data = train) 
sns.countplot(x = "release_day", data = train) 
sns.countplot(x = "release_dayofweek", data = train) 
sns.countplot(x = "release_quarter", data = train)
sns.countplot(x = "has_homepage", data = train) 
sns.countplot(x = "language_en", data = train) 
sns.countplot(x = "rating", data = train)

sns.boxplot(x = "release_month", y = "revenue", data = train)
sns.boxplot(x = "release_dayofweek", y = "revenue", data = train)
sns.boxplot(x = "release_quarter", y = "revenue", data = train)
sns.boxplot(x = "has_homepage", y = "revenue", data = train) 
sns.boxplot(x = "language_en", y = "revenue", data = train) 
sns.boxplot(x = "num_countries", y = "revenue", data = train) 
sns.boxplot(x = "num_genres", y = "revenue", data = train) 

sns.lineplot(x = "release_year", y = "revenue", data = train) 
sns.lineplot(x = "release_year", y = "runtime", data = train) 
sns.lineplot(x = "release_year", y = "popularity", data = train) 
sns.lineplot(x = "release_year", y = "budget", data = train)
sns.lineplot(x = "rating", y = "revenue", data = train)
sns.lineplot(x = "release_year", y = "totalVotes", data = train) 
sns.lineplot(x = "num_cast", y = "revenue", data = train) 
sns.lineplot(x = "num_crew", y = "revenue", data = train)

f, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train.corr())

#####################
# Prepare test data #
#####################

test["release_date"] = test["release_date"].fillna("1/1/00") # fill na in row 828
test[["release_month", "release_day", "release_year"]] = test["release_date"].str.split("/", expand = True).astype(int)
test.loc[test["release_year"] <= 19, "release_year"] += 2000
test.loc[test["release_year"] < 2000, "release_year"] += 1900

release_date = pd.to_datetime(test["release_date"]) 
test["release_dayofweek"] = release_date.dt.dayofweek
test["release_quarter"] = release_date.dt.quarter

test = pd.merge(test, test_extra_feats, how = "left", on = "imdb_id")

test["has_homepage"] = 0  
test.loc[test["homepage"].isnull() == False, "has_homepage"] = 1

test["language_en"] = 0
test.loc[test["original_language"] == "en", "language_en"] = 1

test["has_collection"] = 0
test.loc[test["belongs_to_collection"].isnull() == False, "has_collection"] = 1

test["has_tagline"] = 0
test.loc[test["tagline"].isnull() == False, "has_tagline"] = 1

test["num_countries"] = test["production_countries"].apply(lambda x: len(x) if x != {} else 0) 
test["num_genres"] = test["genres"].apply(lambda x: len(x) if x != {} else 0)
test["num_cast"] = test["cast"].apply(lambda x: len(x) if x != {} else 0)
test["num_crew"] = test["crew"].apply(lambda x: len(x) if x != {} else 0)
test["num_companies"] = test["production_companies"].apply(lambda x: len(x) if x != {} else 0)
test["num_languages"] = test["spoken_languages"].apply(lambda x: len(x) if x != {} else 0)
test["num_keywords"] = test["Keywords"].apply(lambda x: len(x) if x != {} else 0)

test["budget"] = test["budget"] + test["budget"] * 0.018 * (2018 - test["release_year"])
test["budget"] = np.log1p(test["budget"]) 

test["revenue"] = np.log1p(test["revenue"]) # fix skewness

test = test[[
        "budget",
        "popularity",
        "popularity2",
        "runtime",
        "rating",
        "totalVotes",
        "release_month",
        "release_day",
        "release_dayofweek",
        "release_year",
        "release_quarter",
        "has_homepage",
        "has_collection",
        "has_tagline",
        "language_en",
        "num_countries",
        "num_cast",
        "num_genres",
        "num_crew",
        "num_companies",
        "num_languages",
        "num_keywords",
        ]
]

test.isna().sum()    
    
test["runtime"] = test["runtime"].fillna(test["runtime"].mean()) # must impute NA
test["popularity2"] = test["popularity2"].fillna(test["popularity2"].mean())
test["rating"] = test["rating"].fillna(test["rating"].mean())
test["totalVotes"] = test["totalVotes"].fillna(test["totalVotes"].mean())  

x_test = test

####################
## Fitting models ##
####################

# 1. XGBoost
train_xgb = xgb.DMatrix(x_train, y_train)
val_xgb = xgb.DMatrix(x_val, y_val)
test_xgb = xgb.DMatrix(x_test)

params = {
        "objective": "reg:linear",
        "eval_metric": "rmse",
        "max_depth": 5,
        "eta": 0.01,
        "subsample": 0.6,
        "colsample_bytree": 0.7,
        "silent": False
}

model_xgb = xgb.train(
        params,
        train_xgb,
        num_boost_round = 100000,
        evals = [(train_xgb, "train"), (val_xgb, "valid")],
        early_stopping_rounds = 500,
)

preds_xgb = model_xgb.predict(test_xgb, ntree_limit = model_xgb.best_ntree_limit)
                              
xgb.plot_importance(model_xgb)

xgb.to_graphviz(model_xgb, num_trees = model_xgb.best_ntree_limit)

shap_explainer = shap.TreeExplainer(model_xgb) # explainability with SHAP
shap_values = shap_explainer.shap_values(x_train)

shap.force_plot(  # SHAP values for first prediction
        shap_explainer.expected_value,
        shap_values[0, :],
        x_train.iloc[0, :],
        matplotlib = True
)

shap.summary_plot(shap_values, x_train) # feature importance summary

# 2. Catboost
model_cat = CatBoostRegressor(
        iterations = 100000,
        learning_rate = 0.004,
        depth = 5,
        eval_metric = "RMSE",
        colsample_bylevel = 0.8,
        bagging_temperature = 0.2,
        metric_period = None,
        early_stopping_rounds = 200
)

model_cat.fit(
        x_train,
        y_train,
        eval_set = (x_val, y_val),
        use_best_model = True,
        verbose = True
)

preds_cat = model_cat.predict(x_test)

shap_explainer = shap.TreeExplainer(model_cat)
shap_values = shap_explainer.shap_values(x_train)

shap.summary_plot(shap_values, x_train)

# Model averaging
preds = (0.6 * preds_xgb) + (0.4 * preds_cat)

################
## Submission ##
################  

subm = pd.read_csv("./input/sample_submission.csv")

subm["revenue"] = np.expm1(preds)

subm.to_csv("submission.csv", index = False) 
