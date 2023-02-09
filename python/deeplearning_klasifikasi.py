#coba pake beberapa model deep learning
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Permute, Conv1DTranspose, ZeroPadding1D, SimpleRNN, Conv1DTranspose, UpSampling1D, LeakyReLU,  Average, maximum, SeparableConv1D,Multiply, PReLU, Add, SpatialDropout1D, Flatten, Lambda, Concatenate, TimeDistributed, GlobalAveragePooling1D,Activation, BatchNormalization, Conv1D,Input, Dense, LSTM, GRU,Dropout,Flatten, MaxPooling1D, GlobalMaxPool1D
import time

from tensorflow.keras.optimizers import Adam, SGD, Adadelta,RMSprop,Nadam, Adagrad, Adamax

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras import regularizers
import numpy as np
from tensorflow.keras import backend as K
from time import time


import os

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

ACC=tf.keras.metrics.Accuracy
roc=tf.keras.metrics.AUC(
    num_thresholds=200,
    curve='ROC',
    summation_method='interpolation',
    name='ROC'
    
)
auc=tf.keras.metrics.AUC(name='AUC' )


def Irawnet_multi(Xtr1,Xtr2,Ytr,Xts1,Xts2, Yts,timestep,featurex,featurey):
  
    
    

    input_layer1=Input(shape=(timestep,featurex))
    input_layer2=Input(shape=(timestep,featurex))
    
   
    cnn01=Conv1DTranspose(32, 1, padding='same')(input_layer1)
    bn01=BatchNormalization()(cnn01)
    fc01=Activation("tanh")(bn01)
    
    cnn02=Conv1DTranspose(32, 1, padding='same')(input_layer2)
    bn02=BatchNormalization()(cnn02)
    fc02=Activation("tanh")(bn02)
    
    #cross 2 chanel
    # av=Average()([fc01, fc02])
    # c11=Add()([av, fc01])
    # #c11=Activation("tanh")(c1)
    # c2=Add()([av, fc02])
    # c22=Activation("tanh")(c2)
    
   
    cnn2=Conv1D(32, 30, padding='causal', dilation_rate=1)(fc01)
    bn2=BatchNormalization()(cnn2)
    fc2=LeakyReLU(alpha=0.03)(bn2)
    #gabung1=Add()([fc01,fc2])
    #gabung1=Concatenate(axis=2)([fc1,fc2])

    cnn3=Conv1D(32, 30, padding='causal', dilation_rate=2)(fc2)
    bn3=BatchNormalization()(cnn3)
    fc3=LeakyReLU(alpha=0.03)(bn3)
    gabung2=Add()([fc2,fc3])
    #gabung2=Concatenate(axis=2)([gabung1,fc3])

    cnn4=Conv1D(32, 30, padding='causal', dilation_rate=4)(gabung2)
    bn4=BatchNormalization()(cnn4)
    fc4=LeakyReLU(alpha=0.03)(bn4)
    gabung3=Add()([gabung2,fc4])
    #gabung3=Concatenate(axis=2)([gabung2,fc4])
    
    
    cnn5=Conv1D(32, 30, padding='causal', dilation_rate=8)(gabung3)
    bn5=BatchNormalization()(cnn5)
    fc5=LeakyReLU(alpha=0.03)(bn5)
    gabung4=Add()([gabung3,fc5])
    
    
    #gabung4=Concatenate(axis=2)([gabung3,fc5])

    cnn6=Conv1D(32, 30, padding='causal', dilation_rate=16)(gabung4)
    bn6=BatchNormalization()(cnn6)
    fc6=LeakyReLU(alpha=0.03)(bn6)
    gabung5=Add()([gabung4,fc6])
    #gabung5=Concatenate(axis=2)([gabung4,fc6])

    # cnn7=Conv1D(32, 30, padding='causal', dilation_rate=32)(gabung5)
    # bn7=BatchNormalization()(cnn7)
    # fc7=LeakyReLU(alpha=0.03)(bn7)
    # gabung6=Add()([gabung5,fc7])
    
    
    # #gabung6=Concatenate(axis=2)([gabung5,fc7])
    
    # cnn8=Conv1D(32, 30, padding='causal', dilation_rate=64)(gabung6)
    # bn8=BatchNormalization()(cnn8)
    # fc8=LeakyReLU(alpha=0.03)(bn8)
    # gabung7=Add()([gabung6,fc8])
    # #gabung7=Concatenate(axis=2)([gabung6,fc8])
    
    # cnn9=Conv1D(32,30, padding='causal', dilation_rate=128)(gabung7)
    # bn9=BatchNormalization()(cnn9)
    # fc9=LeakyReLU(alpha=0.03)(bn9)
    # gabung8=Add()([gabung7,fc9])
    # #gabung8=Concatenate(axis=2)([gabung7,fc9])
    
    # cnn10=Conv1D(32,30, padding='causal', dilation_rate=256)(gabung8)
    # bn10=BatchNormalization()(cnn10)
    
    # fc10=LeakyReLU(alpha=0.03)(bn10)
    # gabung9=Add()([gabung8,fc10])
    #=========================================right
    rcnn2=Conv1D(32, 30, padding='causal', dilation_rate=1)(fc02)
    rbn2=BatchNormalization()(rcnn2)
    rfc2=LeakyReLU(alpha=0.03)(rbn2)
    #rgabung1=Add()([fc02,rfc2])
    #gabung1=Concatenate(axis=2)([fc1,fc2])

    rcnn3=Conv1D(32, 30, padding='causal', dilation_rate=2)(rfc2)
    rbn3=BatchNormalization()(rcnn3)
    rfc3=LeakyReLU(alpha=0.03)(rbn3)
    rgabung2=Add()([rfc2,rfc3])
    #gabung2=Concatenate(axis=2)([gabung1,fc3])

    rcnn4=Conv1D(32, 30, padding='causal', dilation_rate=4)(rgabung2)
    rbn4=BatchNormalization()(rcnn4)
    rfc4=LeakyReLU(alpha=0.03)(rbn4)
    rgabung3=Add()([rgabung2,rfc4])
    #gabung3=Concatenate(axis=2)([gabung2,fc4])
    
    
    rcnn5=Conv1D(32, 30, padding='causal', dilation_rate=8)(rgabung3)
    rbn5=BatchNormalization()(rcnn5)
    rfc5=LeakyReLU(alpha=0.03)(rbn5)
    rgabung4=Add()([rgabung3,rfc5])
    
    
    #gabung4=Concatenate(axis=2)([gabung3,fc5])

    rcnn6=Conv1D(32, 30, padding='causal', dilation_rate=16)(rgabung4)
    rbn6=BatchNormalization()(rcnn6)
    rfc6=LeakyReLU(alpha=0.03)(rbn6)
    rgabung5=Add()([rgabung4,rfc6])
    # # #gabung5=Concatenate(axis=2)([gabung4,fc6])

    # rcnn7=Conv1D(32, 30, padding='causal', dilation_rate=32)(rgabung5)
    # rbn7=BatchNormalization()(rcnn7)
    # rfc7=LeakyReLU(alpha=0.03)(rbn7)
    # rgabung6=Add()([rgabung5,rfc7])
    
    
    # #gabung6=Concatenate(axis=2)([gabung5,fc7])
    
    # rcnn8=Conv1D(32, 30, padding='causal', dilation_rate=64)(rgabung6)
    # rbn8=BatchNormalization()(rcnn8)
    # rfc8=LeakyReLU(alpha=0.03)(rbn8)
    # rgabung7=Add()([rgabung6,rfc8])
    # #gabung7=Concatenate(axis=2)([gabung6,fc8])
    
    # rcnn9=Conv1D(32,30, padding='causal', dilation_rate=128)(rgabung7)
    # rbn9=BatchNormalization()(rcnn9)
    # rfc9=LeakyReLU(alpha=0.03)(rbn9)
    # rgabung8=Add()([rgabung7,rfc9])
    # #gabung8=Concatenate(axis=2)([gabung7,fc9])
    
    # rcnn10=Conv1D(32,30, padding='causal', dilation_rate=256)(rgabung8)
    # rbn10=BatchNormalization()(rcnn10)
    
    # rfc10=LeakyReLU(alpha=0.03)(rbn10)
    # rgabung9=Add()([rgabung8,rfc10])
    #c=Add()([gabung1, gabung2, gabung3, gabung4, gabung5, gabung6])
    
    c=Concatenate(axis=2)([gabung5, rgabung5, fc01, fc02])
    
    
    out1=Conv1D(featurey,1,activation='sigmoid')(c)
    model=Model(inputs=[input_layer1, input_layer2], outputs=out1)
    model.summary()
    
    
    #es = EarlyStopping(monitor='recall', mode='max', verbose=1, patience=20)
    #mc=ModelCheckpoint(weights_path, monitor='recall', mode='max', verbose=1, save_best_only=True)
  
    opt = RMSprop(learning_rate=0.00001)
    
    model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy', auc, 'Precision', 'Recall', f1_m])
    n_folds=1
    epochs=50
    skor=[]
    prediksi=[]
    start = time()
    
    for i in range(n_folds):
        csvlog=tf.keras.callbacks.CSVLogger('../log/oversampling_multi_rmsprop.log')
        ii=str(i)
        print("k-fold :",i)
        start = time()
        print("start training time: ", start)
        history = model.fit([Xtr1,Xtr2],Ytr, validation_split=0.2, epochs=epochs, batch_size=20, callbacks=[csvlog])
        end=time()
        print("end training time :",end)
        f_hd = '../model_save/oversampling_multi_sigmoid_k' + ii + '.h5'
        model.save(f_hd)
        score=model.evaluate([Xts1,Xts2], Yts)
        skor.append(score)
        pred=model.predict([Xts1, Xts2])
        prediksi.append(pred)
    return history, model,skor,prediksi

def Irawnet_single(Xtr,Ytr,Xts,Yts,timestep,featurex,featurey):
  
    
    

    input_layer1=Input(shape=(timestep,featurex))
    #input_layer2=Input(shape=(timestep,featurex))
    
   
    cnn01=Conv1DTranspose(32, 1, padding='same')(input_layer1)
    bn01=BatchNormalization()(cnn01)
    fc01=Activation("tanh")(bn01)
    
    
    
   
    cnn2=Conv1D(32, 30, padding='causal', dilation_rate=1)(fc01)
    bn2=BatchNormalization()(cnn2)
    fc2=LeakyReLU(alpha=0.03)(bn2)
    gabung1=Add()([fc01,fc2])
    #gabung1=Concatenate(axis=2)([fc1,fc2])

    cnn3=Conv1D(32, 30, padding='causal', dilation_rate=2)(gabung1)
    bn3=BatchNormalization()(cnn3)
    fc3=LeakyReLU(alpha=0.03)(bn3)
    gabung2=Add()([fc2,fc3])
    #gabung2=Concatenate(axis=2)([gabung1,fc3])

    cnn4=Conv1D(32, 30, padding='causal', dilation_rate=4)(gabung2)
    bn4=BatchNormalization()(cnn4)
    fc4=LeakyReLU(alpha=0.03)(bn4)
    gabung3=Add()([gabung2,fc4])
    #gabung3=Concatenate(axis=2)([gabung2,fc4])
    
    
    cnn5=Conv1D(32, 30, padding='causal', dilation_rate=8)(gabung3)
    bn5=BatchNormalization()(cnn5)
    fc5=LeakyReLU(alpha=0.03)(bn5)
    gabung4=Add()([gabung3,fc5])
    
    
    #gabung4=Concatenate(axis=2)([gabung3,fc5])

    cnn6=Conv1D(32, 30, padding='causal', dilation_rate=16)(gabung4)
    bn6=BatchNormalization()(cnn6)
    fc6=LeakyReLU(alpha=0.03)(bn6)
    gabung5=Add()([gabung4,fc6])
    #gabung5=Concatenate(axis=2)([gabung4,fc6])

    con=Concatenate(axis=2)([gabung5,fc01])
    
    
    out1=Conv1D(featurey,1,activation='softmax')(con)
    model=Model(inputs=input_layer1, outputs=out1)
    model.summary()
    
    
    #es = EarlyStopping(monitor='recall', mode='max', verbose=1, patience=20)
    #mc=ModelCheckpoint(weights_path, monitor='recall', mode='max', verbose=1, save_best_only=True)
  
    opt = RMSprop(learning_rate=0.00001)
    
    model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy', auc, 'Precision', 'Recall', f1_m])
    n_folds=5
    epochs=50
    skor_single=[]
    prediksi_single=[]
    start = time()
    i=0
    for i in range(n_folds):
        ii=str(i)
        print("k-fold :",i)
        start = time()
        print("start training time: ", start)
        history = model.fit(Xtr,Ytr, validation_split=0.2, epochs=epochs, batch_size=20)
        end=time()
        print("end training time :",end)
        f_hd = '../model_save/original_mono_k' + ii + '.h5'
        model.save(f_hd)
        score=model.evaluate(Xts, Yts)
        skor_single.append(score)
        pred=model.predict(Xts)
        prediksi_single.append(pred)
    return history, model,skor_single,prediksi_single
    



def DeepWaveNet(Xtr,Ytr,Xts,Yts,timestep,featurex,featurey):
    #DeepWaveNet(lucas,et.al,2018) will be compared with TRN_proposed_method. 
    #preprocessing add zeropadd left and right
    input_layer=Input(shape=(timestep,featurex))
   
   
    identity=Conv1D(128, 1, padding='causal')(input_layer)
   
    cnn1=Conv1D(128, 2, padding='causal', dilation_rate=1)(identity)
    fc1=Activation('tanh')(cnn1)
    fc12=Activation('sigmoid')(cnn1)
    mul1=Multiply()([fc1,fc12])
    
    res1=Add()([identity,mul1])
    
    cnn2=Conv1D(128, 2, padding='causal', dilation_rate=2)(res1)
    fc2=Activation('tanh')(cnn2)
    fc22=Activation('sigmoid')(cnn2)
    mul2=Multiply()([fc2,fc22])
    
    res2=Add()([res1,mul2])
   
    cnn3=Conv1D(128, 2, padding='causal', dilation_rate=4)(res2)
    fc3=Activation('tanh')(cnn3)
    fc32=Activation('sigmoid')(cnn3)
    mul3=Multiply()([fc3,fc32])
    
    res3=Add()([res2,mul3])
    
    cnn4=Conv1D(128, 2, padding='causal', dilation_rate=8)(res3)
    fc4=Activation('tanh')(cnn4)
    fc42=Activation('sigmoid')(cnn4)
    mul4=Multiply()([fc4,fc42])
    
    res4=Add()([res3,mul4])
   
    cnn5=Conv1D(128, 2, padding='causal', dilation_rate=16)(res4)
    fc5=Activation('tanh')(cnn5)
    fc52=Activation('sigmoid')(cnn5)
    mul5=Multiply()([fc5,fc52])
    
    res5=Add()([res4,mul5])
    
    cnn6=Conv1D(128, 2, padding='causal', dilation_rate=32)(res5)
    fc6=Activation('tanh')(cnn6)
    fc62=Activation('sigmoid')(cnn6)
    mul6=Multiply()([fc6,fc62])
    
    res6=Add()([res5,mul6])
    
    cnn7=Conv1D(128, 2, padding='causal', dilation_rate=64)(res6)
    fc7=Activation('tanh')(cnn7)
    fc72=Activation('sigmoid')(cnn7)
    mul7=Multiply()([fc7,fc72])
    
    res7=Add()([res6,mul7])
   
    cnn8=Conv1D(128, 2, padding='causal', dilation_rate=128)(res7)
    fc8=Activation('tanh')(cnn8)
    fc82=Activation('sigmoid')(cnn8)
    mul8=Multiply()([fc8,fc82])
    
    res8=Add()([res7,mul8])
    
    cnn9=Conv1D(128, 2, padding='causal', dilation_rate=256)(res8)
    fc9=Activation('tanh')(cnn9)
    fc92=Activation('sigmoid')(cnn9)
    mul9=Multiply()([fc9,fc92])
    
    res9=Add()([res8,mul9])
    
    cnn10=Conv1D(128, 2, padding='causal', dilation_rate=512)(res9)
    fc10=Activation('tanh')(cnn10)
    fc102=Activation('sigmoid')(cnn10)
    mul10=Multiply()([fc10,fc102])
    
    res10=Add()([res9,mul10])
    
    cnn11=Conv1D(128, 2, padding='causal', dilation_rate=1)(res10)
    fc11=Activation('tanh')(cnn11)
    fc112=Activation('sigmoid')(cnn11)
    mul11=Multiply()([fc11,fc112])
    
    res11=Add()([res10,mul11])
    
    cnn12=Conv1D(128, 2, padding='causal', dilation_rate=2)(res11)
    fc12=Activation('tanh')(cnn12)
    fc122=Activation('sigmoid')(cnn12)
    mul12=Multiply()([fc12,fc122])
    
    res12=Add()([res11,mul12])
   
    cnn13=Conv1D(128, 2, padding='causal', dilation_rate=4)(res12)
    fc13=Activation('tanh')(cnn13)
    fc132=Activation('sigmoid')(cnn13)
    mul13=Multiply()([fc13,fc132])
    
    res13=Add()([res12,mul13])
    
    cnn14=Conv1D(128, 2, padding='causal', dilation_rate=8)(res13)
    fc14=Activation('tanh')(cnn14)
    fc142=Activation('sigmoid')(cnn14)
    mul14=Multiply()([fc14,fc142])
    
    res14=Add()([res13,mul14])
   
    cnn15=Conv1D(128, 2, padding='causal', dilation_rate=16)(res14)
    fc15=Activation('tanh')(cnn15)
    fc152=Activation('sigmoid')(cnn15)
    mul15=Multiply()([fc15,fc152])
    
    res15=Add()([res14,mul15])
    
    cnn16=Conv1D(128, 2, padding='causal', dilation_rate=32)(res15)
    fc16=Activation('tanh')(cnn16)
    fc162=Activation('sigmoid')(cnn16)
    mul16=Multiply()([fc16,fc162])
    
    res16=Add()([res15,mul16])
    
    cnn17=Conv1D(128, 2, padding='causal', dilation_rate=64)(res16)
    fc17=Activation('tanh')(cnn17)
    fc172=Activation('sigmoid')(cnn17)
    mul17=Multiply()([fc17,fc172])
    
    res17=Add()([res16,mul17])
   
    cnn18=Conv1D(128, 2, padding='causal', dilation_rate=128)(res17)
    fc18=Activation('tanh')(cnn18)
    fc182=Activation('sigmoid')(cnn18)
    mul18=Multiply()([fc18,fc182])
    
    res18=Add()([res17,mul18])
    
    cnn19=Conv1D(128, 2, padding='causal', dilation_rate=256)(res18)
    fc19=Activation('tanh')(cnn19)
    fc192=Activation('sigmoid')(cnn19)
    mul19=Multiply()([fc19,fc192])
    
    res19=Add()([res18,mul19])
    
    cnn20=Conv1D(128, 2, padding='causal', dilation_rate=512)(res19)
    fc20=Activation('tanh')(cnn20)
    fc202=Activation('sigmoid')(cnn20)
    mul20=Multiply()([fc20,fc202])
    
 
    
    skipconnection=Add()([mul1, mul2, mul3, mul4, mul5, mul6, mul7,mul8, mul9, mul10, mul11, mul12, mul13, mul14, mul15, mul16, mul17, mul18, mul19, mul20])
    ac1=Activation('relu')(skipconnection)
    ac2=Activation('relu')(ac1)
   
    out=Dense(featurey, activation='sigmoid')(ac2)
    model=Model(inputs=input_layer, outputs=out)
    model.summary()
   # weights_path = os.getcwd() + "\\savemodel1\\Deepwavenet.h5"
    #plotmodel_path=os.getcwd() + "\\gambar1\\Deepwavenet.png"
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
    #mc=ModelCheckpoint(weights_path, monitor='val_recall', mode='max', verbose=1, save_best_only=True)
  
    opt = Adam(learning_rate=0.001)
    #model.compile(opt, loss='mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError()])
    model.compile(opt, loss='binary_crossentropy', metrics=['accuracy', auc, 'Precision', 'Recall', f1_m])
    
    
    history=model.fit(Xtr,Ytr, validation_split=0.2, epochs=50, batch_size=20)
  
    #model.save_weights('/content/drive/MyDrive/projectS3/Transkripsi_Saron/savemodel/skipCon_versilama_4convlayer.h5')
    f_hd = '../model_save/DeepWaveNet_binarycross.h5'
    model.save(f_hd)
    print("Testing")
    testing=model.evaluate(Xts,Yts)
    prediksi=model.predict(Xts)
    return history,model, testing, prediksi

def TCN(Xtr,Ytr,Xts,Yts,timestep,featurex,featurey):
    #TCN bai2020 : An empiracl eva;utaion of Generic Convolution dan RNN of sequence modeling. 
    #perbandingan arsitektur dengan data Nottingdam Music Dataset
    input_layer=Input(shape=(timestep,featurex))
    
   
    cnn1=Conv1D(150,6, padding='same', dilation_rate=1)(input_layer)
    bn1=BatchNormalization()(cnn1)
    fc1=Activation("relu")(bn1)
    d1=Dropout(0.1)(fc1)
   

    cnn2=Conv1D(150,6, padding='same', dilation_rate=2)(d1)
    bn2=BatchNormalization()(cnn2)
    fc2=Activation("relu")(bn2)
    d2=Dropout(0.1)(fc2)
    gabung1=Add()([d1,d2])
   

    cnn3=Conv1D(150,6, padding='same', dilation_rate=4)(gabung1)
    bn3=BatchNormalization()(cnn3)
    fc3=Activation("relu")(bn3)
    d3=Dropout(0.1)(fc3)
    gabung3=Add()([gabung1,d3])

    cnn4=Conv1D(150,6, padding='same', dilation_rate=8)(gabung3)
    bn4=BatchNormalization()(cnn4)
    fc4=Activation("relu")(bn4)
    d4=Dropout(0.1)(fc4)
    gabung3=Add()([gabung3,d4])

    
    out=Dense(featurey, activation='softmax')(gabung3)
    model=Model(inputs=input_layer, outputs=out)
    model.summary()
    
   
  
    opt = RMSprop(learning_rate=0.00001)
    model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy', auc, 'Precision', 'Recall', f1_m])
    
    
    history=model.fit(Xtr,Ytr, validation_split=0.2, epochs=50, batch_size=20)
  
    #model.save_weights('/content/drive/MyDrive/projectS3/Transkripsi_Saron/savemodel/skipCon_versilama_4convlayer.h5')
    f_hd = '../model_save/TCN.h5'
    model.save(f_hd)
    print("Testing")
    testing=model.evaluate(Xts,Yts)
    prediksi=model.predict(Xts)
    return history,model, testing, prediksi






