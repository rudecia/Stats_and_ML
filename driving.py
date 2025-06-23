import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import root_mean_squared_error, r2_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns


#Driving Test Data

# Gender: Gender identity of applicant.
# Age Group: Factor of three levels indicating applicants' age group.
# Race: Applicant's race or ethnicity.
# Training: Factor of three levels indicating applicants' previous training before the test.
# Signals: Assessment score of proper signalling, lane changes, and staying within lanes.
# Yield: Assessment score of right of way to other vehicles and pedestrians.
# Speed Control: Measure of ability to maintain appropriate speed based on traffic and road conditions.
# Night Drive: Performance score out of 100 in simulated or actual night driving conditions.
# Road Signs: Score on applicant's knowledge indicating familiarity and correct interpretation of road signs.
# Steer Control: Score of the applicant's ability to control the vehicle under normal and stressful conditions.
# Mirror Usage: Score on proper and consistent use of mirrors during various manoeuvres.
# Confidence: Evaluator's subjective score on how confidently the applicant handled driving tasks.
# Parking: Evaluation score for parallel, angle, and perpendicular parking tasks.
# Theory Test: Score out of 100 on the in-car theoretical assessment covering traffic laws, road signs, and general driving theory.
# Reactions: Factor of three levels indicating applicants' response to driving scenarios.
# Qualified: Indicator for whether applicant qualifies for a driver's license.

driving_data = pd.read_csv('/Users/rudeciabernard/Downloads/Drivers License Data.csv').dropna().set_index('Applicant ID')



#driving:
#numerically encoding categorical variables
driving_data['Gender'] = driving_data['Gender'].replace(['Male', 'Female'], [1, 2])
driving_data['Age Group'] = driving_data['Age Group'].replace(['Teenager', 'Young Adult', 'Middle Age'], [1, 2, 3])
driving_data['Race'] = driving_data['Race'].replace(['Other', 'Black', 'White'], [1, 2, 3])
driving_data['Training'] = driving_data['Training'].replace(['Advanced', 'Basic'], [2, 1])
driving_data['Reactions'] = driving_data['Reactions'].replace(['Average', 'Slow', 'Fast'], [2, 1, 3])
driving_data['Qualified'] = driving_data['Qualified'].replace(['Yes', 'No'], [1, 0])




sns.heatmap(driving_data.corr(), cmap = 'coolwarm', annot= True)
plt.title('Correlation Heat Map for Driving Test Performance')
plt.show()



X = driving_data.drop('Qualified', axis = 1)
Y = driving_data[['Qualified']]

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size= 0.75, random_state= 42, shuffle= True)



#Logistic Regression

d_log_reg = LogisticRegressionCV(cv = 10)
d_log_reg.fit(x_train, y_train)
log_preds = d_log_reg.predict(x_test)


#Decision Tree
d_tree = DecisionTreeClassifier()
d_tree.fit(x_train, y_train)
tree_preds = d_tree.predict(x_test)


#Random Forest Classifier
d_forest = RandomForestClassifier()
d_forest.fit(x_train, y_train)
for_preds = d_forest.predict(x_test)



#test + compare models
print(f'logistic accuracy is {accuracy_score(y_test, log_preds)}; r2 score is {r2_score(y_test, log_preds)}; rmse is {root_mean_squared_error(y_test, log_preds)}')
print(f'tree accuracy is {accuracy_score(y_test, tree_preds)}; r2 score is {r2_score(y_test, tree_preds)}; rmse is {root_mean_squared_error(y_test, tree_preds)}')
print(f'forest accuracy is {accuracy_score(y_test, for_preds)}; r2 score is {r2_score(y_test, for_preds)}; rmse is {root_mean_squared_error(y_test, for_preds)}')





#Dimensionality Reduction w/ Principal Components Analysis

prince = PCA(n_components= 3)
X_pca = prince.fit_transform(X)


pca_X_and_Y = pd.concat([pd.DataFrame(X_pca), Y.reset_index()[['Qualified']]], axis = 1)
passed_test =  pca_X_and_Y[pca_X_and_Y['Qualified'] == 1]
failed_test = pca_X_and_Y[pca_X_and_Y['Qualified'] == 0]



#Let's see how well the components represent passes and failures
#Passes are plotted in green; failures are plotted in red


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(passed_test[0], passed_test[1], passed_test[2], c = 'g')
ax.scatter(failed_test[0], failed_test[1], failed_test[2], c = 'r')

ax.set_title('Driving Data PCA 3D Representation')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')

plt.show()

#The plot shows that the first component captures the passes and failures pretty well.




#Modeling with the PCA transformed data
x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, Y ,train_size= 0.75, random_state= 42, shuffle = True )



#Logistic Regression w/ pca
d_log_pca =  LogisticRegressionCV(cv = 10)
d_log_pca.fit(x_train_pca, y_train)
log_preds_pca = d_log_pca.predict(x_test_pca)


#Decision Tree w/pca
d_tree_pca = DecisionTreeClassifier()
d_tree_pca.fit(x_train_pca, y_train)
tree_preds_pca = d_tree_pca.predict(x_test_pca)


#Random Forest Classifier w/pca
d_forest_pca = RandomForestClassifier()
d_forest_pca.fit(x_train_pca, y_train)
for_preds_pca = d_forest_pca.predict(x_test_pca)


print(f'logistic + pca accuracy is {accuracy_score(y_test, log_preds_pca)}; r2 score is {r2_score(y_test, log_preds_pca)}; rmse is {root_mean_squared_error(y_test, log_preds_pca)}')
print(f'tree accuracy + pca is {accuracy_score(y_test, tree_preds_pca)}; r2 score is {r2_score(y_test, tree_preds_pca)}; rmse is {root_mean_squared_error(y_test, tree_preds_pca)}')
print(f'forest accuracy + pca is {accuracy_score(y_test, for_preds_pca)}; r2 score is {r2_score(y_test, for_preds_pca)}; rmse is {root_mean_squared_error(y_test, for_preds_pca)}')


#There isn't a large difference in performance b/t 2 - 3 principal components and using all 15 covariates











pass


