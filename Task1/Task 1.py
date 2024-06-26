import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('updated_dating.csv')

df = df.dropna(subset=['career'])

scaler = StandardScaler()
df[['income']] = scaler.fit_transform(df[['income']])

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df[['age', 'like', 'income']] = imputer.fit_transform(df[['age', 'like', 'income']])

X = df[['attr', 'fun', 'age', 'like', 'intel', 'amb', 'gender', 'sinc', 'income']]
y = df['dec']

x_train_1, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM model 1
svm_1 = SVC(kernel='rbf')
svm_1.fit(x_train_1, y_train)


y_pred = svm_1.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy for the SVM classifier: {accuracy:.2f}")


plt.figure(figsize=(10, 6))


scatter = plt.scatter(x_train_1['attr'], x_train_1['fun'], c=y_train, cmap='coolwarm', edgecolors='k', s=100)
plt.xlabel('Attraction')
plt.ylabel('Fun')
plt.title('Decision Boundary and Scatter Plot of attr vs fun for Dating Dataset')

# SVM model 2
svm_2 = SVC()
svm_2.fit(x_train_1[['attr', 'fun']], y_train)

DecisionBoundaryDisplay.from_estimator(
    svm_2,
    x_train_1[['attr', 'fun']],
    response_method="predict",
    cmap=plt.cm.coolwarm,
    alpha=0.2,
    ax=plt.gca()
)

plt.legend(handles=scatter.legend_elements()[0], labels=['Negative Decision', 'Positive Decision'])

plt.show()
