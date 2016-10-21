## import libraries

import h5py
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Read raw data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print('Dim train', train.shape)
print('Dim test', test.shape)

# Scan features
target = [value for value in list(train) if not value.endswith("loss")]
continuous = [value for value in list(train) if value.startswith("cont")]
categorical = [value for value in list(train) if value.startswith("cat")]

n_train = train.shape[0]
y = train['loss']
id_train = train['id']
id_test = test['id']

# Create scaler
scaler = StandardScaler()
scaler.fit(train[continuous])

# scale the continuous data
data_cont = pd.concat((train[continuous], test[continuous]), axis=0)
data_cont = pd.DataFrame(scaler.transform(data_cont), columns=continuous)
data_cont.reset_index(inplace=True)

data_cat = pd.concat((train[categorical], test[categorical]), axis=0)
data_cat.reset_index(inplace=True)
del train, test

# create dummies for categorical
for f in categorical:
    dummies = pd.get_dummies(data_cat[f], prefix=f)
    for col in dummies.columns:
        data_cont[col] = dummies[col]

# omit index
features = data_cont.columns[1:]

with h5py.File('data/data.h5', 'w') as hf:
    hf.create_dataset('features', data=list(features))
    hf.create_dataset('X_train', data=data_cont[features].values[:n_train, :])
    hf.create_dataset('y_train', data=y.values)
    hf.create_dataset('id_train', data=id_train.values)
    hf.create_dataset('X_test', data=data_cont[features].values[n_train:, :])
    hf.create_dataset('id_test', data=id_test.values)

print('Dim X_train', data_cont.values[:n_train, 1:].shape)
print('Dim y_train', y.shape)
print('Dim id_train', id_train.shape)
print('Dim X_test', data_cont.values[n_train:, 1:].shape)
print('Dim id_test', id_test.shape)
