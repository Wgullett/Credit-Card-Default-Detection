import pandas as pd
#import sklearn as sk
from xgboost import XGBClassifier
from matplotlib import pyplot

df = pd.read_csv("default_credit_cards.csv", index_col= 0)

print(df)

df = df.iloc[1:]

Y = df[df.columns.values[23]]
Y = Y.astype(int)


X = df.drop(df.columns.values[23], axis=1)
X = X.astype(int)

total = X.shape[0]
num_test = int(total*0.8)

X_train = X.iloc[:num_test]
Y_train = Y.iloc[:num_test]

X_test = X.iloc[num_test:].reset_index()
X_test = X_test.drop(X_test.columns.values[0], axis=1)

Y_test = Y.iloc[num_test:].reset_index()
Y_test = Y_test.drop(Y_test.columns.values[0], axis=1)


model = XGBClassifier(objective='binary:logistic')

model.fit(X_train, Y_train)

predictions = model.predict(X_test)

actual = list(Y_test['Y'])

total = len(actual)

sum = 0
for i in range(total):
    if predictions[i] == actual[i]:
        sum+=1

print(sum/total)

print(model.feature_importances_)
