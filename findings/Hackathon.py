
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn


# In[ ]:


hw = pd.read_csv("training_data_utf8.csv")
val = pd.read_csv("validation_data.csv")


# In[ ]:


datum = val.transaction_ym.apply(pd.to_datetime,format="%Y-%m")
share_neu = val.plot_area * val.plot_share

val["paidshare"] = share_neu
val["t_month"] = datum.dt.month
val["t_year"] = datum.dt.year

val_clean = val[["id", "price_per_m2","t_month","t_year","paidshare","property_type","cadastral","contract_type","protection_zone","building_ban"]]


# In[ ]:


datum = hw.transaction_ym.apply(pd.to_datetime,format="%Y-%m")
share_neu = hw.plot_area * hw.plot_share

hw["paidshare"] = share_neu
hw["t_month"] = datum.dt.month
hw["t_year"] = datum.dt.year

hw_clean = hw[["price_per_m2","t_month","t_year","paidshare","property_type","cadastral","contract_type","protection_zone","building_ban"]]
hw_clean = hw_clean.dropna()


# In[ ]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(hw_clean, test_size=0.2, random_state=42)


# In[ ]:


data = train_set.drop("price_per_m2", axis=1)
labels = train_set["price_per_m2"].copy()


# In[ ]:


num_attribs = ["paidshare", "t_year", "t_month"]
#cat_attribs = ["property_type", "cadastral", "contract_type", "protection_zone", "building_ban"]


# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# In[ ]:


from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline

class LabelBinarizerPipelineFriendly(LabelBinarizer):
    def fit(self, X, y=None):
        """this would allow us to fit the model based on the X input."""
        super(LabelBinarizerPipelineFriendly, self).fit(X)
    def transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).transform(X)

    def fit_transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).fit(X).transform(X)


# In[ ]:


prop_pipeline = Pipeline([
    ("selector", DataFrameSelector("property_type")),
    ("label_binarizer", LabelBinarizerPipelineFriendly()),
])

cada_pipeline = Pipeline([
    ("selector", DataFrameSelector("cadastral")),
    ("label_binarizer", LabelBinarizerPipelineFriendly()),
])
cont_pipeline = Pipeline([
    ("selector", DataFrameSelector("contract_type")),
    ("label_binarizer", LabelBinarizerPipelineFriendly()),
])
prot_pipeline = Pipeline([
    ("selector", DataFrameSelector("protection_zone")),
    ("label_binarizer", LabelBinarizerPipelineFriendly()),
])
buil_pipeline = Pipeline([
    ("selector", DataFrameSelector("building_ban")),
    ("label_binarizer", LabelBinarizerPipelineFriendly()),
])


# encoder = LabelBinarizer
# cat_data = data[cat_attribs].copy()
# cat_l = encoder.fit_transform(cat_data)
# cat_l.head()

# In[ ]:


num_pipeline = Pipeline([
    ("selector", DataFrameSelector(num_attribs))
])


# In[ ]:


from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("property_type_transform", prop_pipeline),
    ("cadastral_transform", cada_pipeline),
    ("contract_type_transform", cont_pipeline),
    ("protection_zone_transform", prot_pipeline),
    ("building_type", buil_pipeline)
])


# In[ ]:


data_prepared = full_pipeline.fit_transform(data)
#val_prepared = full_pipeline.fit_transform(val_clean)


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

forest_reg =  RandomForestRegressor()
forest_reg.fit(data_prepared, labels)


# In[ ]:


scores = cross_val_score(forest_reg, data_prepared, labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)


# In[ ]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation", scores.std())
          
display_scores(forest_rmse_scores)


# from sklearn.model_selection import GridSearchCV
# 
# param_grid = [
#     {"n_estimators": [3,10,30], "max_features": [2,4,6,8,10,12,14]},
#     {"bootstrap": [False], "n_estimators": [20,50], "max_features": [2,3,4]}
# ]
# 
# forest_reg = RandomForestRegressor()
# 
# grid_search = GridSearchCV(forest_reg, param_grid, cv =5, scoring="neg_mean_squared_error")
# 
# grid_search.fit(data_prepared, labels)

# grid_search.best_params_

# cvres = grid_search.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print (np.sqrt(-mean_score), params)

# In[ ]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(data_prepared, labels)


# scores = cross_val_score(lin_reg, data_prepared, labels, scoring="neg_mean_squared_error", cv=10)
# lin_rmse_scores = np.sqrt(-scores)

# display_scores(lin_rmse_scores)

# In[ ]:


test_data = test_set.drop("price_per_m2", axis=1)
test_labels = test_set["price_per_m2"].copy()
test_prepared = full_pipeline.transform(test_data)


# In[ ]:


from sklearn.metrics import mean_squared_error

predictlin = lin_reg.predict(test_prepared)
predictforest = forest_reg.predict(test_prepared)

lin_final_mse = mean_squared_error(test_labels, predictlin)
for_final_mse = mean_squared_error(test_labels, predictforest)

lin_rmse = np.sqrt(lin_final_mse)
forest_rmse = np.sqrt(for_final_mse)

print (lin_rmse, forest_rmse)


# In[ ]:


val_prepared = full_pipeline.transform(val_clean)


# In[ ]:


predictions = forest_reg.predict(val_prepared)


# In[ ]:


val_clean["res"] = predictions
val_clean[["id","res"]].set_index("id").to_csv("prediction_Team5_DarkDataDestroyer420.csv")

