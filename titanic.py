import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

pd.set_option('display.max_rows', 300)

# INITIAL DATA PROCESSING/INSPECTION
def preprocess(df):
	df = df.copy()
	df['Cabin_class'] = df['Cabin'].str.extract('([a-zA-Z])', expand=False)
	df['Cabin_class'] = df['Cabin_class'].replace(['D', 'E', 'B'], 'D/E/B')
	df['Cabin_class'] = df['Cabin_class'].replace(['F', 'C'], 'F/C')
	df['Cabin_class'] = df['Cabin_class'].replace(['G', 'A'], 'G/A')
	df['Cabin_class'] = df['Cabin_class'].replace('T', np.nan)
	df['Cabin_class'] = df['Cabin_class'].replace(np.nan, 'NS')
	df.loc[df['SibSp'] > 1, 'SibSp'] = 1
	df.loc[df['Parch'] > 2, 'Parch'] = 2
	df['Relatives'] = 0
	df.loc[(df['SibSp'] > 0) | (df['Parch'] > 0), 'Relatives'] = 1
	df['Age**2'] = df['Age']**2
	df['Age_cat'] = pd.cut(df['Age'], [0, 5, 15, 59, 80], right=True)
	df['Fare**2'] = df['Fare']**2
	df['Fare_cat'] = pd.cut(df['Fare'], [0, 8.5, 26, 80, 513], right=True, include_lowest=True)
	for dummy in ['Pclass', 'Sex', 'Embarked', 'Cabin_class', 'Age_cat', 'Fare_cat',]:
		df = pd.concat([df, pd.get_dummies(df[dummy], dummy)], axis=1)
	return df

def survivedbycat(col):
	print(train[[col, 'Survived']].groupby(col).agg(['mean', 'count']).sort_values(by=('Survived', 'mean'), ascending=False))

train = preprocess(pd.read_csv('train.csv'))
print(train.info())
survivedbycat('Pclass')
survivedbycat('Sex')
survivedbycat('SibSp')
survivedbycat('Parch')
survivedbycat('Relatives')
print(train[['Fare', 'Survived']].groupby('Survived').agg(['mean', 'count']))
survivedbycat('Fare_cat')
survivedbycat('Cabin_class')
survivedbycat('Embarked')
print(train['Age'].describe())
print(train['Age'].value_counts().sort_index().cumsum()/714)
print(train[['Age', 'Survived']].groupby('Age').agg(['mean', 'count']))
survivedbycat('Age_cat')

# FIT LOGISTIC REGRESSION MODEL
class DataFrameSelector(BaseEstimator, TransformerMixin):
	def __init__(self, age_cat=False, fare_cat=False, relatives_dum=False):
		self.age_cat = age_cat
		self.fare_cat = fare_cat
		self.relatives_dum = relatives_dum

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
			'Cabin_class_D/E/B', 'Cabin_class_F/C', 'Cabin_class_G/A', 'Cabin_class_NS', 'Embarked_S', 'Embarked_C', 'Embarked_Q',]
		if self.age_cat:
			columns += ['Age_cat_(0, 5]', 'Age_cat_(5, 15]', 'Age_cat_(15, 59]', 'Age_cat_(59, 80]',]
		else:
			columns += ['Age', 'Age**2',]
		if self.fare_cat:
			columns += ['Fare_cat_(-0.001, 8.5]', 'Fare_cat_(8.5, 26.0]', 'Fare_cat_(26.0, 80.0]', 'Fare_cat_(80.0, 513.0]']
		else:
			columns += ['Fare', 'Fare**2',]
		if self.relatives_dum:
			columns += ['Relatives',]
		else:
			columns += ['SibSp', 'Parch',]
		return X[columns].values

pipe = Pipeline([
	('selector', DataFrameSelector()),
	('imputer', Imputer(strategy = 'median')),
	('logreg', LogisticRegression(random_state=42)),
])

param_grid = {
	'selector__age_cat': [False, True],
	'selector__fare_cat': [False, True],
	'selector__relatives_dum': [False, True],
}
grid_search = GridSearchCV(pipe, param_grid,
	scoring={'precision': make_scorer(precision_score), 'recall': make_scorer(recall_score), 'f1': make_scorer(f1_score), 'accuracy': make_scorer(accuracy_score)}, refit='accuracy')
grid_search.fit(train, train['Survived'])
print('Best parameters: {0}'.format(grid_search.best_params_))
print('Best accuracy score: {0}'.format(grid_search.best_score_))
cvres = grid_search.cv_results_
print('{0:100} {1:>10} {2:>10} {3:>10} {4:>10}'.format('Parameters', 'Precision', 'Recall', 'F1', 'Accuracy'))
for precision, recall, f1, accuracy, params in zip(cvres['mean_test_precision'], cvres['mean_test_recall'], cvres['mean_test_f1'], cvres['mean_test_accuracy'], cvres['params']):
	print('{0:100} {1:>10.5f} {2:>10.5f} {3:>10.5f} {4:>10.5f}'.format(str(params), precision, recall, f1, accuracy))
pred = cross_val_predict(grid_search.best_estimator_, train, train['Survived'])
print(confusion_matrix(train['Survived'], pred))
print('Precision: ', precision_score(train['Survived'], pred))
print('Recall: ', recall_score(train['Survived'], pred))
print('F1: ', f1_score(train['Survived'], pred))
print('Accuracy: ', accuracy_score(train['Survived'], pred))

# CHECK MODEL ON TEST DATA
test = preprocess(pd.read_csv('test.csv'))
test_pred = grid_search.best_estimator_.predict(test)
submission = test[['PassengerId']].copy()
submission['Survived'] = test_pred
submission.to_csv('submission.csv', index=False)

