# Credit Card Fraud Detection

[Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) is a highly unbalanced
dataset consisting of 492 fraud and 284,807 normal transactions. Every record in data frame has 30 
features which are:
  - _Time_: the number of seconds elapsed between each transaction and the first transaction;
  - _Amount_: the transaction amount;
  - _V1_, _V2_, _..._, _V28_: generated by PCA features of transaction.
  
---
  
Residual neural network was built as classifier. The NN consists of 6 fully connected layers and 2
shortcuts connections. Here is its structure:  

<pre>
___________________________________________________________________
Layer           Output Shape        Activation        Connected to
===================================================================
input           [(None, 29)]        
___________________________________________________________________
dense_0          (None, 64)           ReLu             input
___________________________________________________________________
dense_1          (None, 64)           ReLU             dense_0
___________________________________________________________________
add_0            (None, 64)                            dense_1
                                                       dense_0
___________________________________________________________________
dense_2          (None, 64)           ReLU             add_0
___________________________________________________________________
dense_3          (None, 29)           ReLU             dense_2
___________________________________________________________________
dense_4          (None, 29)           ReLU             dense_3
___________________________________________________________________
add_1            (None, 29)                            dense_4
                                                       dense_3
                                                       input
___________________________________________________________________
dense_5          (None, 29)           ReLU             add_1
___________________________________________________________________
dense_6          (None, 1)            Sigmoid          dense_5
===================================================================
</pre>

The model is saved in `model.h5` file.  
Test accuracy of the model is 0.93, ROC AUC is 0.97 and F1-Score is 0.93.

---

The project is completed using [DVC](https://dvc.org/). There are 3 stages in total:  
  - __stages/split.dvc__ - Pre-process the dataset: removes column _Time_, creates _RobustScaler_
  and use it on column _Amount_, split dataset into train and test with test_size=0.2;
  - __stages/train.dvc__ - Trains the model and saves it to _data/model.h5_ for DVC and to 
  _model.h5_ for Git;
  - __stages/evaluate.dvc__ - Evaluates the model via test dataset. Outputs accuracy, ROC AUC score
   and F1-Score. Saves all metrics to _metrics.txt_, so later you can view them using `dvd metrics
   show`.

There is a visualisation of whole pipeline of __stages/evaluate.dvc__ below:  
<pre>
                 +-------------------------+
                 | data\creditcard.csv.dvc |
                 +-------------------------+
                               *
                               *
                               *
                     +------------------+
                     | stages\split.dvc |
                     +------------------+
                   ***                  ****
               ****                         ***
             **                                ****
+------------------+                               **
| stages\train.dvc |                           ****
+------------------+                        ***
                   ***                  ****
                      ****          ****
                          **      **
                   +---------------------+
                   | stages\evaluate.dvc |
                   +---------------------+
</pre> 

To reproduce the model and the evaluation process use: `dvc repro stages/evaluate.dvc`.
  