import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay

df = pd.read_csv("updated_dating.csv")


columns_with_nan = ['age', 'like', 'fun', 'attr', 'intel', 'sinc', 'amb', 'income']
df[columns_with_nan] = df[columns_with_nan].apply(lambda col: col.fillna(col.mean()))

# 1st Model (all features except Career) :
X_1 = df.drop(columns=['career', 'dec']).values
y = df['dec'].values
X_train_1, X_test_1, y_train, y_test = train_test_split(X_1, y, test_size=0.2, random_state=42)
model_1 = SVC(kernel='rbf')
model_1.fit(X_train_1, y_train)
accuracy_1 = model_1.score(X_test_1, y_test)
print(f"Model Accuracy (All Except Career): {accuracy_1}")


#2nd Model (only fun and attr in consideration)
X_2 = df[['fun', 'attr']].values
X_train_2, X_test_2, y_train, y_test = train_test_split(X_2, y, test_size=0.2, random_state=42)
model_2 = SVC(kernel='rbf')
model_2.fit(X_train_2, y_train)


plt.figure(figsize=(8, 6))
DecisionBoundaryDisplay.from_estimator(
    model_2, X_2, response_method="predict", cmap=plt.cm.coolwarm, alpha=0.8, ax=plt.gca()
)

plt.scatter(X_2[y == 0, 0], X_2[y == 0, 1], color='blue', label='Unmatched')
plt.scatter(X_2[y == 1, 0], X_2[y == 1, 1], color='red', label='Matched')

plt.xlabel('Attractiveness')
plt.ylabel('Fun')
plt.title('SVM Decision Boundary with Attractiveness and Fun')
plt.legend(title='Match')
plt.tight_layout()
plt.show()
