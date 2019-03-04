import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#reading data
df = pd.read_csv('KNN_Project_Data')
df.head()
df.info()

#exploratory data analysis
sns.set()
sns.pairplot(df,hue='TARGET CLASS',kind='scatter',diag_kind='hist')
plt.show()

#standardizing variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
data_df = pd.DataFrame(scaled_features,columns=df.columns[:-1])
data_df.head()

#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],test_size=0.30,random_state=101)

#knn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

#choosing k value
error_rate = []

for i in range(1,41):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,41),error_rate,color='blue', linestyle='--', marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

#knn with new value
knn = KNeighborsClassifier(n_neighbors=31)
knn.fit(X_train,y_train)
pred_31 = knn.predict(X_test)
print(classification_report(y_test,pred_31))