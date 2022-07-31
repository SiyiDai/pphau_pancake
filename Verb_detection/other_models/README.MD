### Classification Reports for Models Tryouts

#### Autoencoder - 2xLSTM+2XLSTM

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 10, 128)           130560    
                                                                 
 dropout (Dropout)           (None, 10, 128)           0         
                                                                 
 lstm_1 (LSTM)               (None, 10, 64)            49408     
                                                                 
 lstm_2 (LSTM)               (None, 10, 64)            33024     
                                                                 
 lstm_3 (LSTM)               (None, 10, 128)           98816     
                                                                 
 dropout_1 (Dropout)         (None, 10, 128)           0         
                                                                 
 time_distributed (TimeDistr  (None, 10, 126)          16254     
 ibuted)                                                         
                                                                 
 flatten (Flatten)           (None, 1260)              0         
                                                                 
 dense_1 (Dense)             (None, 128)               161408    
                                                                 
 dense_2 (Dense)             (None, 12)                1548      
                                                                 
=================================================================
Total params: 491,018
Trainable params: 491,018
Non-trainable params: 0
_________________________________________________________________
```

```
              precision    recall  f1-score   support

       close       0.96      0.77      0.85        60
    decorate       0.97      0.94      0.95       288
        flip       0.81      0.89      0.85        80
 move_object       0.78      0.75      0.77        76
        open       0.96      0.91      0.94        80
     pick_up       0.92      0.90      0.91       431
        pour       0.97      0.92      0.95       165
    put_down       0.85      0.83      0.84       209
       screw       0.93      0.98      0.95       333
      shovel       0.81      0.81      0.81       209
     squeeze       0.91      0.98      0.94       171
       other       0.99      0.99      0.99      5423

    accuracy                           0.97      7525
   macro avg       0.90      0.89      0.90      7525
weighted avg       0.97      0.97      0.97      7525


```
1/1 [==============================] - 0s 6ms/step

The current action is: close


#### Encoder with 2 x LSTM

```
Model: "Encoder with 2 x LSTM"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 0_LSTM (LSTM)               (None, 10, 128)           130560    
                                                                 
 1_Dropout (Dropout)         (None, 10, 128)           0         
                                                                 
 2_LSTM (LSTM)               (None, 10, 64)            49408     
                                                                 
 7_Flatten (Flatten)         (None, 640)               0         
                                                                 
 8_Dense (Dense)             (None, 128)               82048     
                                                                 
 9_Dense (Dense)             (None, 12)                1548      
                                                                 
=================================================================
Total params: 263,564
Trainable params: 263,564
Non-trainable params: 0
_________________________________________________________________
```
```
             precision    recall  f1-score   support

       close       0.91      0.83      0.87        60
    decorate       0.97      0.96      0.97       288
        flip       0.83      0.89      0.86        80
 move_object       0.80      0.89      0.84        76
        open       0.90      1.00      0.95        80
     pick_up       0.92      0.95      0.93       431
        pour       0.98      0.91      0.94       165
    put_down       0.92      0.80      0.85       209
       screw       0.95      0.97      0.96       333
      shovel       0.86      0.78      0.82       209
     squeeze       0.95      0.96      0.96       171
       other       0.99      0.99      0.99      5423

    accuracy                           0.97      7525
   macro avg       0.91      0.91      0.91      7525
weighted avg       0.97      0.97      0.97      7525

```
1/1 [==============================] - 0s 5ms/step

The current action is: close




#### LSTM 128

```
Model: "LSTM_128"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 0_LSTM (LSTM)               (None, 10, 128)           130560    
                                                                 
 1_Dropout (Dropout)         (None, 10, 128)           0         
                                                                 
 7_Flatten (Flatten)         (None, 1280)              0         
                                                                 
 8_Dense (Dense)             (None, 128)               163968    
                                                                 
 9_Dense (Dense)             (None, 12)                1548      
                                                                 
=================================================================
Total params: 296,076
Trainable params: 296,076
Non-trainable params: 0
_________________________________________________________________
```

```
              precision    recall  f1-score   support

       close       0.91      0.87      0.89        60
    decorate       0.96      0.95      0.96       288
        flip       0.87      0.89      0.88        80
 move_object       0.96      0.84      0.90        76
        open       0.82      0.97      0.89        80
     pick_up       0.94      0.94      0.94       431
        pour       0.98      0.95      0.96       165
    put_down       0.89      0.85      0.87       209
       screw       0.96      0.95      0.95       333
      shovel       0.87      0.77      0.81       209
     squeeze       0.91      0.96      0.93       171
       other       0.99      0.99      0.99      5423

    accuracy                           0.97      7525
   macro avg       0.92      0.91      0.91      7525
weighted avg       0.97      0.97      0.97      7525
```

1/1 [==============================] - 0s 4ms/step

The current action is: close


#### LSTM 64
```
Model: "LSTM_64"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 0_LSTM (LSTM)               (None, 10, 64)            48896     
                                                                 
 1_Dropout (Dropout)         (None, 10, 64)            0         
                                                                 
 2_LSTM (LSTM)               (None, 10, 64)            33024     
                                                                 
 7_Flatten (Flatten)         (None, 640)               0         
                                                                 
 8_Dense (Dense)             (None, 128)               82048     
                                                                 
 9_Dense (Dense)             (None, 12)                1548      
                                                                 
=================================================================
Total params: 165,516
Trainable params: 165,516
Non-trainable params: 0
_________________________________________________________________
```

```
              precision    recall  f1-score   support

       close       0.78      0.93      0.85        60
    decorate       0.95      0.96      0.96       288
        flip       0.89      0.70      0.78        80
 move_object       0.81      0.80      0.81        76
        open       0.94      0.84      0.89        80
     pick_up       0.97      0.84      0.90       431
        pour       0.95      0.98      0.97       165
    put_down       0.74      0.75      0.74       209
       screw       0.91      0.97      0.94       333
      shovel       0.86      0.75      0.80       209
     squeeze       0.94      0.96      0.95       171
       other       0.98      0.99      0.99      5423

    accuracy                           0.96      7525
   macro avg       0.89      0.87      0.88      7525
weighted avg       0.96      0.96      0.96      7525

```

1/1 [==============================] - 0s 5ms/step

The current action is: close
