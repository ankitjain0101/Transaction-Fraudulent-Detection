import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,roc_auc_score,precision_recall_curve,matthews_corrcoef
from sklearn.metrics import classification_report,accuracy_score,recall_score,precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

credit = pd.read_csv('E:/College/Analytics/Predictive/Assignment 4/creditcard.csv')
credit.shape
credit.dtypes
credit.isnull().any().any()
credit[["Time","Amount","Class"]].describe()
credit['Class'] = pd.Categorical(credit.Class)
print(credit['Class'].value_counts())
sns.countplot(credit['Class'])

x = credit.drop(["Class"], axis=1).values 
y = credit["Class"].cat.codes

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=615)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.2,random_state=615)
x_train.shape
x_test.shape

# Logistic Regression
logr = LogisticRegression(random_state=96)
logr.fit(x_train, y_train)
pred_log= logr.predict(x_test)
pd.crosstab(y_test,pred_log, rownames=['True'], colnames=['Predicted'], margins=True)
matthews_corrcoef(y_test,pred_log)
precision_score(y_test,pred_log)
recall_score(y_test,pred_log)
accuracy_score(y_test,pred_log)

# Random Forest
par_grid= {"min_samples_leaf" : [1,2,3,4],"criterion":["gini","entropy"],"max_depth":[3,5,7],
           "max_features":[6,7,8]}

rf=RandomForestClassifier(random_state=96)
grid_search= GridSearchCV(rf,param_grid=par_grid,cv=5)
grid_search.fit(x_val, y_val)
print(grid_search.best_score_)
print(grid_search.best_params_)

rf=RandomForestClassifier(criterion='gini', min_samples_leaf=1,max_features=6,max_depth=5)
rf.fit(x_train, y_train)
pred= rf.predict(x_test)
print(rf.score(x_train, y_train))
confusion_matrix(y_test,pred)
pd.crosstab(y_test,pred, rownames=['True'], colnames=['Predicted'], margins=True)
matthews_corrcoef(y_test,pred)
precision_score(y_test,pred)
recall_score(y_test,pred)
accuracy_score(y_test,pred)
roc_auc_score(y_test,pred)
print(classification_report(y_test,pred))

y_pred_prob = rf.predict_proba(x_test)[:,1]
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot(precision, recall)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
plt.show()

x1 = credit.columns.tolist()
del x1[30]

tmp=pd.DataFrame({'Feature': x1,'Feature importance': rf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
tmp1 = tmp.head(5)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp1)
s.set_title('Top 5 Features importance',fontsize=20)
