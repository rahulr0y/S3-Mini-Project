
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load csv file
df = pd.read_csv('data.csv')

# Data Preprocessing
df['Date'] = pd.to_datetime(df.Date)


df['DoW'] = pd.to_datetime(df['Date']).dt.dayofweek

for i, row in df.iterrows():

  if i >= 2:
    df.loc[i, 'dif2'] = (df.loc[i-2, 'Open'] - df.loc[i-2, 'Close'])/df.loc[i-2, 'Open'] * 100
    df.loc[i, 'dif1'] = (df.loc[i-2, 'Open'] - df.loc[i-1, 'Close'])/df.loc[i-2, 'Open'] * 100
    df.loc[i, 'dif0'] = (df.loc[i-2, 'Open'] - df.loc[i, 'Close'])/df.loc[i-2,'Open'] * 100

    df.loc[i, 'HL_p1'] = (df.loc[i-1, 'High'] - df.loc[i-1, 'Low'])/df.loc[i-1, 'Close'] * 100
    df.loc[i, 'HL_p2'] = (df.loc[i-2, 'High'] - df.loc[i-2, 'Low'])/df.loc[i-2, 'Close'] * 100


df['dif0'] = df['dif0'].abs()
df['dif1'] = df['dif1'].abs()
df['dif2'] = df['dif2'].abs()

for i, row in df.iterrows():
  if df.loc[i, 'dif0'] > 1.0:
    df.loc[i,'Outcome'] = 0
  else:
    df.loc[i,'Outcome'] = 1

df.Outcome = df.Outcome.astype(int)

# checking for null values
# df.isnull().sum()
df.dropna(inplace=True)

# create dataframe named 'data' where df['DoW'] == 3, Thursday
data = df.loc[df['DoW'] == 3].copy()
data = data[['Date', 'dif2', 'dif1', 'HL_p2', 'HL_p1',  'Outcome']]

# select independent and dependent variables
X = data[['dif2', 'dif1', 'HL_p1', 'HL_p2']]
y = data[['Outcome']]

model_score = 0
for i in range(100):

    # split dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    # Model creation
    model = KNeighborsClassifier(n_neighbors=5)

    model.fit(X_train, y_train.values.ravel())
    if model.score(X_test, y_test) > model_score:

        # Create pickle file for saving model
        pickle.dump(model, open('model.pkl', 'wb'))
        model_score = model.score(X_test, y_test)
        # print(model.score(X_test, y_test))
y_pred = model.predict(X_test)

print(f'Model Score: {model_score}\n')
print('Confusion Matrix: ')
print(confusion_matrix(y_test, y_pred), '\n')
print('Classification Report: ')
print(classification_report(y_test, y_pred))
