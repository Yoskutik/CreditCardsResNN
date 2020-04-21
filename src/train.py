from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import pandas as pd


df = pd.read_csv('data/train.csv')
X = df.iloc[:, df.columns != 'Class'].values
y = df.Class.values

input_0 = layers.Input(shape=(29,))
dense_0 = layers.Dense(64, activation='relu')(input_0)
dense_1 = layers.Dense(64, activation='relu')(dense_0)
dense_1 = layers.Dense(64, activation='relu')(
    layers.Add()([dense_1, dense_0])
)
dense_0 = layers.Dense(29, activation='relu')(dense_1)
dense_1 = layers.Dense(29, activation='relu')(dense_0)
dense_1 = layers.Dense(29, activation='relu')(
    layers.Add()([dense_1, dense_0, input_0])
)
out = layers.Dense(1, activation='sigmoid')(dense_1)


model = models.Model(input_0, out)
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(5e-4))
history = model.fit(X, y, epochs=300, verbose=0)

y_pred = model.predict(X)
print(f'Train accuracy:  {accuracy_score(y, np.where(y_pred > 0.5, 1, 0)):0.5f}')
print(f'Train ROC AUC:   {roc_auc_score(y, y_pred):0.5f}')

model.save('model.h5')
model.save('data/model.h5')
