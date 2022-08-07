import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("BankNote_Authentication.csv")
print(df.head())

y = df['class']
X = df[['variance','skewness','curtosis','entropy']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)




LR = LogisticRegression() 
LR.fit(X_train,y_train)

#X_test = np.reshape(X_test, (-1,1))
#Y_test = np.reshape(y_test, (-1,1))


y_prediction = LR.predict(X_test)

predicted_values = []

for i in y_prediction:
    if i == 0:
        predicted_values.append("Forged")
    else:
        predicted_values.append("Autorized")
        
actual_values = []

for i in y_test:
    if i == 0:
        actual_values.append("Forged")
    else:
        actual_values.append("Authorized")
        
labels = ["Authorized", "Forged"]
cm = confusion_matrix(actual_values, predicted_values, labels)

ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax)

ax.set_xlabel("Forged")
ax.set_ylabel("Authorized")
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)
plt.show()





