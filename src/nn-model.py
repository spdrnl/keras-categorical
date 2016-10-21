import h5py
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Merge
from keras.models import Sequential


def get_feature_lanes(features):
    # define model input lanes
    lanes = []
    current_concept = ''
    current_features = []
    for f in features:
        concept = 'cont' if f.startswith('cont') else f.split('_')[0]
        if concept != current_concept:
            if current_features != []:
                lanes.append(current_features)
                # print len(current_features), current_features
                current_features = []
            current_concept = concept
        current_features.append(f)
    lanes.append(current_features)
    # print len(current_features), current_features
    # duplets
    duplets = [lane[0] for lane in [lane for lane in lanes if len(lane) == 2]]
    lanes = [lane for lane in lanes if len(lane) > 2]
    lanes[0].extend(duplets)
    return lanes


def get_data_lanes(df):
    feature_lanes = get_feature_lanes(df.columns)
    data_lanes = []
    for lane in feature_lanes:
        #print lane
        data_lanes.append(df[lane].values)
    return data_lanes


def get_lane_model(df):
    feature_lanes = get_feature_lanes(df.columns)
    model_lanes = []

    for index, feature_lane in enumerate(feature_lanes):
        model = Sequential()
        if index == 0:
            model.add(Dense(len(feature_lane), input_dim=len(feature_lane), input_shape=len(feature_lane), init='he_normal'))
        else:
            model.add(Dense(1, input_dim=len(feature_lane), init='he_normal'))
        model_lanes.append(model)

    merged = Merge(model_lanes, mode='concat')

    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dense(200, activation='relu'))
    final_model.add(Dropout(0.2))
    final_model.add(Dense(150, activation='relu'))
    final_model.add(Dropout(0.2))
    final_model.add(Dense(50, activation='relu'))
    final_model.add(Dense(1, init='he_normal'))
    final_model.compile(loss='mae', optimizer='adadelta')
    return final_model


## read data
with h5py.File('data/data.h5', 'r') as hf:
    # print('List of arrays in this file: \n', hf.keys())
    features = [x.decode() for x in hf.get('features')]
    X_train = np.array(hf.get('X_train'))
    y_train = np.array(hf.get('y_train'))
    id_train = np.array(hf.get('id_train'))
    X_test = np.array(hf.get('X_test'))
    id_test = np.array(hf.get('id_test'))

df = pd.DataFrame(data=X_train, index=range(X_train.shape[0]), columns=features)
print len(df.columns), len(features)
lane_model = get_lane_model(df)
data_lanes = get_data_lanes(df)

nb_epoch = 300
batch_size = 10000
history = lane_model.fit(data_lanes, y_train, validation_split=0.2, shuffle=True, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)
print history.history
