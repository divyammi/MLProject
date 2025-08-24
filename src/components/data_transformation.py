# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# warnings.filterwarnings('ignore')


# df = pd.read_csv("C:/Users/divyamm/Downloads/Learning/Python_with_Data_Science/VS_Code/Day48(MLProject)/notebook/data/stud.csv")
# print(df.head())

# # We will be predicting MAth score on the basis oof all other independent features

# '''Preparing X and Y values'''
# X = df.drop(columns=['math_score'] , axis=1)

# y = df['math_score']


# # since the categorical features doesnot have many features so we can go with one hot encoding and then oonce all are numerical we can perform standardization

# # Create Column Transformer with 3 types of transformers
# num_feratures = X.select_dtypes(exclude="object").columns
# cat_feratures = X.select_dtypes(include="object").columns

# from sklearn.preprocessing import OneHotEncoder,StandardScaler
# from sklearn.compose import ColumnTransformer

# num_transformer = StandardScaler()
# cat_transformer = OneHotEncoder()

# #COmbining the both OneHotEncoder and StandardScaler wiill be done by ColumnTransformer

# preprcessor = ColumnTransformer(
#     [
#         ("OneHotEncoder", cat_transformer, cat_feratures),
#         ("StandardScaler", num_transformer,num_feratures)
#     ]
# )

# X=preprcessor.fit_transform(X)
# print(X)


# #Seperate the data in train test split
# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# #Create an Evaluation function to give all metrics after model selection
# from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
# def evaluate_model(true, predicted):
#     mae = mean_absolute_error(true, predicted)
#     mse = mean_squared_error(true, predicted),
#     rmse  = np.sqrt(mean_squared_error(true, predicted)),
#     r2_scores = r2_score(true, predicted)
#     return mae,mse,rmse,r2_scores


# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
# from sklearn.svm import SVR
# from sklearn.linear_model import LinearRegression, Ridge,Lasso
# from sklearn.model_selection import RandomizedSearchCV
# from catboost import CatBoostRegressor
# from xgboost import XGBRegressor
# import warnings


# models = {
#     "Linear Regression": LinearRegression(),
#     "Lasso": Lasso(),
#     "Ridge": Ridge(),
#     "K-Neighbors Regressor": KNeighborsRegressor(),
#     "Decision Tree": DecisionTreeRegressor(),
#     "Random Forest Regressor": RandomForestRegressor(),
#     "XGBRegressor": XGBRegressor(), 
#     "CatBoosting Regressor": CatBoostRegressor(verbose=False),
#     "AdaBoost Regressor": AdaBoostRegressor()
# }

# model_list = []
# r2_sores_list = []

# for i in range(len(list(models))):
#     model = list(models.values())[i]
#     model.fit(X_train,y_train)  #training each mopdel one by one

#     #make predictions
#     y_train_pred = model.predict(X_train)
#     y_test_pred = model.predict(X_test)


#     #Evaluate train test data on the method evaluate
#     model_train_mae, model_train_mse, model_train_rmse, model_train_r2_scores = evaluate_model(y_train, y_train_pred)
#     model_test_mae, model_test_mse, model_test_rmse, model_test_r2_scores = evaluate_model(y_test, y_test_pred)


#     #Printing Everything
#     print(list(models.keys())[i])
#     model_list.append(list(models.keys())[i])

#     print("Model performance training set")
#     print("- Root Mean Squared Error: ", model_train_rmse)
#     print("- Mean Absolute Error: ",model_train_mae)
#     print("- R2 Score: ",model_train_r2_scores)

#     print('----------------------------------')
    
#     print('Model performance for Test set')
#     print("- Root Mean Squared Error: ",model_test_rmse)
#     print("- Mean Absolute Error: ",model_test_mae)
#     print("- R2 Score: ", model_test_r2_scores)
#     r2_sores_list.append(model_test_r2_scores)
#     print('='*35)
#     print('\n')


# result = pd.DataFrame(list(zip(model_list, r2_sores_list)), columns=['Model Name', 'R2_Score']).sort_values(by=["R2_Score"],ascending=False)

# print(result)

# '''As Linear regression model performed best oin this data we will apply linear regression'''

# lin_model = LinearRegression(fit_intercept=True)
# lin_model = lin_model.fit(X_train, y_train)
# y_pred = lin_model.predict(X_test)
# score = r2_score(y_test, y_pred)*100
# print(" Accuracy of the model is %.2f" %score)


# plt.scatter(y_test,y_pred)
# plt.xlabel('Actual')
# plt.ylabel('Predicted')


# sns.regplot(x=y_test,y=y_pred,ci=None,color ='red')



# pred_df=pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_pred,'Difference':y_test-y_pred})
# print(pred_df)




















































