import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import r_regression
import numpy as np
import seaborn as sns

fetal_health = pd.read_csv('/Users/rudeciabernard/Downloads/fetal_health.csv')


#Predictor Correlations w/ Fetal Health
correlation_matrix  = fetal_health.corr()
fh_r_vals = correlation_matrix.iloc[21, :].sort_values(ascending = False)

#Correlation Matrix


#Examining the Most Significant Correlates to Fetal Health
states = ['Normal', 'Suspect', 'Pathological']
fh_group_data = fetal_health.groupby(by = 'fetal_health').mean().reset_index()

plt.bar(x = fh_group_data['fetal_health'], height= fh_group_data['accelerations'], tick_label = states)
plt.title('Avg Accelerations per Second')
plt.show()

plt.bar(x = fh_group_data['fetal_health'], height= fh_group_data['prolongued_decelerations'], tick_label = states)
plt.title('Avg Prolonged Decelerations per Second')
plt.show()

plt.bar(x = fh_group_data['fetal_health'], height= fh_group_data['abnormal_short_term_variability'], tick_label = states)
plt.title('Avg Pct of Time w/ Abnormal Short-Term Variability in HR')
plt.show()

plt.bar(x = fh_group_data['fetal_health'], height= fh_group_data['percentage_of_time_with_abnormal_long_term_variability'], tick_label = states)
plt.title('Avg Pct of Time w/ Abnormal Long-Term Variability')
plt.show()



#Predicting Fetal Health Classification with Tree Classifiers

#Decision Tree Classifier
X = fetal_health.drop(['fetal_health'], axis= 1).copy()
y = fetal_health[['fetal_health']].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 33, shuffle= True, train_size= 0.75)


dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)


print(f'decision tree rmse is {root_mean_squared_error(y_test, dtc.predict(X_test))}')
print(f'decision tree accuracy is {accuracy_score(y_test, dtc.predict(X_test))}')
print(f'decision tree max depth is {dtc.tree_.max_depth}')



#Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

print(f'random forest accuracy is {accuracy_score(y_test, rfc.predict(X_test))}')

# the random forest's accuracy is about the same as the decision tree, if not very slightly higher





pass


