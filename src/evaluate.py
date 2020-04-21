from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tensorflow.keras import models
import numpy as np
import pandas as pd
import json


df = pd.read_csv('data/test.csv')
X_test = df.iloc[:, df.columns != 'Class'].values
y_test = df.Class.values

model = models.load_model('model.h5')
y_pred = model.predict(X_test)
y_pred_step = np.where(model.predict(X_test) > 0.5, 1, 0)

acc = accuracy_score(y_test, y_pred_step)
print(f'Test accuracy:   {acc:0.5f}')
roc_auc = roc_auc_score(y_test, y_pred)
print(f'Test ROC AUC:    {roc_auc:0.5f}')
f1 = f1_score(y_test, y_pred_step)
print(f'Test F1-Score:   {f1:0.5f}')

with open('metrics.txt', 'w') as metrics_file:
    json.dump({
        'Accuracy': f'{acc:0.5f}',
        'ROC AUC': f'{roc_auc:0.5f}',
        'F1-Score': f'{f1:0.5f}',
    }, metrics_file)
