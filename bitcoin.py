import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
 
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('bitcoin.csv')
print(df.head())
print(df.shape)
print(df.describe())
print(df.info())

plt.figure(figsize=(15, 5))
plt.plot(df['Close'])
plt.title('Bitcoin Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()

print("Check if data is null:")
print(df.isnull().sum())

features = ['Open', 'High', 'Low', 'Close']
 
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
  plt.subplot(2,2,i+1)
  sb.distplot(df[col])
plt.show()
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
  range_min = 0.0
  range_max = 10000.0
  filtered_data = [x for x in df[col] if range_min <= x <= range_max]
  plt.subplot(2,2,i+1)
  sb.distplot(filtered_data)
plt.show()
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
  plt.subplot(2,2,i+1)
  ax = sb.boxplot(df[col])
  q1, median, q3 = df[col].quantile([0.25, 0.5, 0.75])
  label_text=f"25% : {q1:.2f}   median : {median:.2f}   75% : {q3:.2f}"
  plt.text(20000, -0.35, label_text, fontsize=12)
  IQR = q3 - q1
  k = 1.5  # Adjust this value if needed
  lower_fence = q1 - k * IQR
  upper_fence = q3 + k * IQR
  label_fence=f"Fence line : {upper_fence:.2f}"
  plt.text(20000, -0.25, label_fence, fontsize=12)
plt.show()

splitted = df['Date'].str.split('-', expand=True)
 
df['year'] = splitted[0].astype('int')
df['month'] = splitted[1].astype('int')
df['day'] = splitted[2].astype('int')
 
print(df.head())

data_grouped = df.groupby('year').mean()
plt.subplots(figsize=(20,10))
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
  plt.subplot(2,2,i+1)
  data_grouped[col].plot.bar()
  dataLabel = f"{col}"
  plt.text(3, 39000, dataLabel, fontsize=12)
plt.show()

df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
print(df.head())
df['open-close']  = df['Open'] - df['Close']
df['low-high']  = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 0, 1)
plt.pie(df['target'].value_counts().values, 
        labels=["Goes down", "Goes up"], autopct='%1.1f%%')
plt.show()

plt.figure(figsize=(10, 10))
 
# As our concern is with the highly
# correlated features only so, we will visualize
# our heatmap as per that criteria only.
sb.heatmap(df.corr() > 0.8, annot=True, cbar=False)
plt.show()

features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']
 
scaler = StandardScaler()
features = scaler.fit_transform(features)
 
X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)

models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]
 
for i in range(3):
  models[i].fit(X_train, Y_train)
  print(f'{models[i]} : ')
  print('Training Accuracy : ', metrics.roc_auc_score(Y_train, models[i].predict_proba(X_train)[:,1]))
  print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, models[i].predict_proba(X_valid)[:,1]))
  print()
print('\n\n0 : Goes up')
print('1 : Goes down')
metrics.plot_confusion_matrix(models[0], X_valid, Y_valid)
plt.show()

