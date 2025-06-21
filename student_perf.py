import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import accuracy_score, r2_score, root_mean_squared_error
from sklearn.cluster import k_means


student_perf = pd.read_csv('/Users/rudeciabernard/Downloads/StudentPerformanceFactors.csv').dropna()


X = student_perf.drop(['Exam_Score'], axis= 1).copy()
X['Extracurricular_Activities'] = X['Extracurricular_Activities'].replace(['No', 'Yes'], [0, 1])
X['Motivation_Level'] = X['Motivation_Level'].replace(['Low', 'Medium', 'High'], [0, 1, 2])
X['Access_to_Resources'] = X['Access_to_Resources'].replace(['Low', 'Medium', 'High'], [0, 1, 2])
X['Internet_Access'] = X['Internet_Access'].replace(['No', 'Yes'], [0, 1])
X['Teacher_Quality'] = X['Teacher_Quality'].replace(['Low', 'Medium', 'High'], [0, 1, 2])
X['School_Type'] = X['School_Type'].replace(['Public', 'Private'], [0, 1])
X['Peer_Influence'] = X['Peer_Influence'].replace(['Negative','Neutral', 'Positive'], [0, 1, 2])
X['Learning_Disabilities'] = X['Learning_Disabilities'].replace(['No', 'Yes'], [0, 1])
X['Parental_Education_Level'] = X['Parental_Education_Level'].replace(['High School', 'College', 'Postgraduate'], [0, 1, 2])
X['Gender'] = X['Gender'].replace(['Male', 'Female'], [0, 1])
X['Distance_from_Home'] = X['Distance_from_Home'].replace(['Near', 'Moderate', 'Far'], [0, 1, 2])
X['Parental_Involvement'] = X['Parental_Involvement'].replace(['Low', 'Medium', 'High'], [0, 1, 2])
X['Family_Income'] = X['Family_Income'].replace(['Low', 'Medium', 'High'], [0, 1, 2])


y = student_perf[['Exam_Score']].copy()

nsp = pd.concat([X, y], axis= 1)

#nsp = 'numerical student performance'

#Exploratory Analyses:
#Correlation Matrix and Heat Map:
nsp_corr_matrix = nsp.corr()
sns.heatmap(nsp_corr_matrix, cmap = 'coolwarm', annot= True)
plt.title('Correlation Heat Map for Student Performance')
plt.show()

#Hours spent studying seems to be the most strongly correlated with performance.



plt.scatter(X['Hours_Studied'], y)
plt.title('Test Score vs. Hours Studied')
plt.xlabel('Hours Studied')
plt.ylabel('Test Score')
plt.show()

#There seems to be two groups of people:
#1) those whose time spent studying has an essentially linear relationship w/ their score (tend to be below 80)
    # reminiscent of people who work really hard but lack natural aptitude; they are the vast majority of the dataset
#2) those whose time spent studying is only loosely related to their test score, if at all (tend to be above 80)
    # perhaps these are the people who are naturally gifted; they make up a really small portion of the dataset



study_scores = student_perf[['Hours_Studied', 'Exam_Score']].copy().reset_index(drop = True)


above_80 = study_scores[study_scores['Exam_Score'] >= 80]
below_80 = study_scores[study_scores['Exam_Score'] < 80]

#Best-fit Line for Non-Outliers
lowx = np.linspace(0, 60)
lowy = [0.29379632 * x + 61.21188454306003 for x in lowx]

plt.scatter(above_80['Hours_Studied'], above_80['Exam_Score'], c= 'g')
plt.scatter(below_80['Hours_Studied'], below_80['Exam_Score'], c = 'r')
plt.plot(lowx, lowy, c = 'b')
plt.title('Scores vs Hrs Studied')
plt.xlabel('Hrs Studied')
plt.ylabel('Scores')
plt.show()



#How well can we predict student scores based on the given info?

#Predicting Student Scores
#Whole Dataset:

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, random_state= 44, shuffle = True)
lr = LinearRegression()
lr.fit(X_train, y_train)

rlp = np.round(lr.predict(X_test)) #rlp = rounded linear predictions

#Linear Regression Performance
print(f'linear regression rmse is {root_mean_squared_error(y_test, rlp)}')
print(f'linear regression r2 is {r2_score(y_test, rlp)}')
print( f'the accuracy of linear regression is {accuracy_score(y_test, rlp)}')

# # pass

#Lasso Performance (L1)
lasso = Lasso()
lasso.fit(X_train, y_train)
rlap = np.round(lasso.predict(X_test)) #rlap = rounded lasso predictions

print(f'lasso regression w/ cross validation rmse is {root_mean_squared_error(y_test, rlap)}')
print(f'lasso regression w/ cross validation r2 is {r2_score(y_test, rlap)}')
print( f'the accuracy of lasso regression w/ cross validation is {accuracy_score(y_test, rlap)}')


#Ridge Regression Performance (L2)
ridge = Ridge()
ridge.fit(X_train, y_train)
rrp = np.round(ridge.predict(X_test)) #rrp = rounded ridge predictions

print(f'ridge regression rmse is {root_mean_squared_error(y_test, rrp)}')
print(f'ridge regression  r2 is {r2_score(y_test, rrp)}')
print( f'the accuracy of ridge regression is {accuracy_score(y_test, rrp)}')




#Overall, it seems like the linear models top out at around 80% accuuracy. 
#Even with cross-validation (see below), the performance does not meaningfully improve


# grid_search_ridge = Ridge()
# #Grid Search
# param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
#               'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg','saga']}

# grid_search = GridSearchCV(grid_search_ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(X_train, y_train)

# print(grid_search.best_estimator_)


# grid_search_ridge = grid_search.best_estimator_
# grid_search_ridge.fit(X_train, y_train)
# rgsp = np.round(grid_search_ridge.predict(X_test)) #rgsp = rounded grid search predictions

# print(f'ridge regression w/ grid search cv has an rmse of {root_mean_squared_error(y_test, rgsp)}')
# print(f'ridge regression w/ grid search cv has an r2 of {r2_score(y_test, rgsp)}')
# print( f'the accuracy of ridge regression w/ grid search cv is {accuracy_score(y_test, rgsp)}')

# #Elastic Net w/ Grid Search CV

# elastic = ElasticNet()

# param2 = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
#           'l1_ratio': [0.2, 0.4, 0.6, 0.8, 0.9],
#           'selection': ['cyclic', 'random']}

# grid_search_en = GridSearchCV(elastic, param2, cv = 5, scoring ='neg_root_mean_squared_error')
# grid_search_en.fit(X_train, y_train)

# print(grid_search_en.best_estimator_)


# elastic = grid_search_en.best_estimator_
# elastic.fit(X_train, y_train)
# renp = np.round(elastic.predict(X_test))

# print(f'elastic net regression w/ grid search cv has an rmse of {root_mean_squared_error(y_test, renp)}')
# print(f'elastic net regression w/ grid search cv has an r2 of {r2_score(y_test, renp)}')
# print( f'the accuracy of elastic net regression w/ grid search cv is {accuracy_score(y_test, renp)}')



#Will these models perform better when the high scorers are removed?
#DATASET W/ OUTLIERS REMOVED:
low_score_nsp = nsp[nsp['Exam_Score'] < 80]

lsX = low_score_nsp.drop(['Exam_Score'], axis = 1).copy()
lsY = low_score_nsp[['Exam_Score']].copy()


lsX_train, lsX_test, lsY_train, lsY_test = train_test_split(lsX, lsY, train_size= 0.8, random_state= 42, shuffle = True)



lr2 = LinearRegression()
lr2.fit(lsX_train, lsY_train)
rounded_pred = np.round(lr2.predict(lsX_test))

print(f'linear regression has an rmse of {root_mean_squared_error(lsY_test, rounded_pred)}')
print(f'linear regression has an r2 of {r2_score(lsY_test, rounded_pred)}')
print(f'the accuracy of the linear regression is {accuracy_score(lsY_test, rounded_pred)}')



#The accuracy went up significantly--- from 80 to 90 percent
pass