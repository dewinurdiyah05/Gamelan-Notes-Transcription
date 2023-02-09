# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 14:32:51 2021

@author: asus
"""
#from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTE
from baca import *
from deeplearning_klasifikasi import *
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
path='../data'

total,wavs,waktu,fs,grup,data, chanel1, chanel2=bacasignal(path)




targetsaron='../data'

csvnotasi,timetarget,Xnumber,p,number=bacatarget(targetsaron)
csvnotubah=rescalenotasi(csvnotasi)

index,amplitudo1,amplitudo2, arrnote,wav1,wav2=matchnotasi(chanel1,chanel2,waktu, timetarget,csvnotubah)

#susun data
data1=[]
data2=[]
notas=[]
l=len(amplitudo1)
for k in range(l):
    j=len(amplitudo1[k])
    for s in range(j-1):
        data1.append(amplitudo1[k][s])
        data2.append(amplitudo2[k][s])
        notas.append(arrnote[k][s])
        
data_satu=np.array(data1)
data_dua=np.array(data2)
note=np.array(notas)

data_1=np.resize(data_satu,[np.size(data_satu,0),1])
data_2=np.resize(data_dua,[np.size(data_dua,0),1])
nada=np.resize(note,[np.size(note,0),1])

data=np.concatenate([data_1, data_2], axis=1)
oversample = SMOTE(sampling_strategy='not majority')
dataover, noteover=oversample.fit_resample(data, nada)
#cek distribusi class setelah over sampling
n=np.size(data,0)
ix=np.resize(nada,[n])
jum0=jum1=jum2=jum3=jum4=jum5=jum6=jum7=0


for i in range(n-1):
        if (ix[i:i+1]==0):
            jum0=jum0+1      
        if (ix[i:i+1]==1):
            jum1=jum1+1
        if (ix[i:i+1]==2):
            jum2=jum2+1
        if (ix[i:i+1]==3):
            jum3=jum3+1
        if(ix[i:i+1]==4):
            jum4=jum4+1
        if(ix[i:i+1]==5):
            jum5=jum5+1
        if(ix[i:i+1]==6):
            jum6=jum6+1
        if(ix[i:i+1]==7):
            jum7=jum7+1
maks=jum2
jum0=(jum0/maks)*100  
jum1=(jum1/maks)*100 
jum2=(jum2/maks)*100
jum3=(jum3/maks)*100      
jum4=(jum4/maks)*100
jum5=(jum5/maks)*100
jum6=(jum6/maks)*100
jum7=(jum7/maks)*100    

            
xx=[jum0, jum1, jum2, jum3, jum4, jum5, jum6, jum7]     
plt.figure(figsize=(8,5))             
plt.plot([0, 1, 2, 3, 4, 5, 6, 7], xx, color='green', linestyle='dashed', linewidth = 2,
         marker='o', markerfacecolor='blue', markersize=10)

plt.axis([0, 7, 0, 100])
plt.xlabel("Categorical note")
plt.ylabel("Number of each categorical note (%) ")
plt.savefig("original.jpg", dpi=1000)
plt.show()



Xtrain=[]
Xvalid=[]
Xtest=[]
Ytrain=[]
Yvalid=[]
Ytest=[]

Xtr=[]
Ytr=[]
Xval=[]
Yval=[]
Xts=[]
Yts=[]
Xall=[]

win=2756

hop=0

amp1=data_satu
amp2=data_dua
overlap=win-hop
X1,X2,Yawal,Yasli,Yencoder,bagi,jumlah,j,m1,m2=Bagiper10msdetik2(amp1, amp2,note,win,hop)

Xtr1,Xtr2,Ytr1, Ytr2,Xts1,Xts2,Yts1,Yts2=splitdata1(X1,X2, Yencoder)
 
Ytr, Yts, timestep,featurex,featurey=reshapedata(Ytr1, Yts1, win)
 
#training IRawNet multi channel with oversampled data from SMOTE result
history_oversample_multi,model_oversample_multi, skor, prediksi=Irawnet_multi(Xtr1,Xtr2,Ytr,Xts1,Xts2, Yts,timestep,featurex,featurey)

#averaging amplitude of left and right channel
Xtr=(Xtr1+Xtr2)/2
Xts=(Xts1+Xts2)/2

#training IRawNet single channel with oversample data

history_oversample_single,model_oversample_single,skor_ov_single,prediksi_ov_single =Irawnet_single(Xtr,Ytr,Xts,Yts,timestep,featurex,featurey)

##training with original data original data
oX1,oX2,oYawal,Yoasli,oYencoder,obagi,ojumlah,oj,om1,om2=Bagiper10msdetik2(data_1, data_2,nada,win,hop)

oXtr1,oXtr2,oYtr1, oYtr2,oXts1,oXts2,oYts1,oYts2=splitdata1(oX1,oX2, oYencoder)
 
oYtr, oYts, otimestep,ofeaturex,ofeaturey=reshapedata(oYtr1, oYts1, win)
 
history_original_multi,model_original_multi, skor_original, prediksi_original=Irawnet_multi(oXtr1,oXtr2,oYtr,oXts1,oXts2, oYts,timestep,featurex,featurey)


#=============================Explore Filter dimmension from oversampling data with IRawNet multi channel
history_64,model_64, skor_64, pred_64=Irawnet_multi(Xtr1,Xtr2,Ytr,Xts1,Xts2, Yts,timestep,featurex,featurey)

history_128,model_128, skor_128, pred_128=Irawnet_multi(Xtr1,Xtr2,Ytr,Xts1,Xts2, Yts,timestep,featurex,featurey)
#====================explore kernel siz e
history_k3,model_k3, skor_3, prediksi_3=Irawnet_multi(Xtr1,Xtr2,Ytr,Xts1,Xts2, Yts,timestep,featurex,featurey)
history_k150,model_150, skor_150, prediksi_150=Irawnet_multi(Xtr1,Xtr2,Ytr,Xts1,Xts2, Yts,timestep,featurex,featurey)

## effect of alpha value of leaky relu activation function
history_a0_003,model_a0_003, skor_a0_003, pred_a0_003=Irawnet_multi(Xtr1,Xtr2,Ytr,Xts1,Xts2, Yts,timestep,featurex,featurey)
history_a0_3,model_a0_3, skor_a0_3, pred_a0_3=Irawnet_multi(Xtr1,Xtr2,Ytr,Xts1,Xts2, Yts,timestep,featurex,featurey)

# effect of optimizer type--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

history_adam,model_adam, skor_adam, pred_adam=Irawnet_multi(Xtr1,Xtr2,Ytr,Xts1,Xts2, Yts,timestep,featurex,featurey)
history_rmsprop,model_rmsprop, skor_rmsprop, pred_rmsprop=Irawnet_multi(Xtr1,Xtr2,Ytr,Xts1,Xts2, Yts,timestep,featurex,featurey)
Editplothasil(history_oversample_multi, history_rmsprop, history_adam)



#effect number of convolutional block
history_2cb,model_2cb,skor_2cb, pred_2cb =Irawnet_multi(Xtr1,Xtr2,Ytr,Xts1,Xts2, Yts,timestep,featurex,featurey)
history_9cb,model_9cb, skor_9cb, pred_10cb=Irawnet_multi(Xtr1,Xtr2,Ytr,Xts1,Xts2, Yts,timestep,featurex,featurey)

#based on number of parameters the best convolutional block is 5 conv block. 

#ivestigate fuzed feature layers
history_NoTrans_NoFuzed,model_NoTrans_NoFuzed, skor_no_fuzed, pred_no_fuzed=Irawnet_multi(Xtr1,Xtr2,Ytr,Xts1,Xts2, Yts,timestep,featurex,featurey)

#comparative performance with Deep WaveNet and TCN
history_DeepWavenet,model_DeepWavenet, skor_DeepWavenet, pred_DeepWavenet=DeepWaveNet(Xtr,Ytr,Xts,Yts,timestep,featurex,featurey)
history_TCN,model_TCN, skor_TCN, pred_TCN=TCN(Xtr,Ytr,Xts,Yts,timestep,featurex,featurey)

##=============load model
import tensorflow as tf
proposed= tf.keras.models.load_model('../model_save/oversampling_multi_k4.h5', compile = True)
model_DeepWavenet=tf.keras.models.load_model('../model_save/DeepWaveNet.h5', compile = False)
model_TCN=tf.keras.models.load_model('../model_save/TCN.h5', compile = False)
model_monochanel=tf.keras.models.load_model('../model_save/oversampling_mono_k4.h5', compile = False)
proposed.summary()



pred_proposed=proposed.predict([Xts1, Xts2])
pred_deepwavnet=model_DeepWavenet.predict(Xts)
pred_TCN=model_TCN.predict(Xts)
pred_mono=model_monochanel(Xts)
       


timestep=2756
hsl_proposed=np.zeros([436,timestep])
hsl_deepwavenet=np.zeros([436,timestep])
hsl_tcn=np.zeros([436,timestep])
hsl_mono=np.zeros([436,timestep])
asli=np.zeros([436,timestep])
for i in range(436):
    for j in range(timestep-1):
        note_proposed=np.argmax(pred_proposed[i:i+1,j:j+1,:])
        # note_deepwavenet=np.argmax(pred_deepwavnet[i:i+1,j:j+1,:])
        # note_tcn=np.argmax(pred_TCN[i:i+1,j:j+1,:])
        noteasli=np.argmax(Yts[i:i+1,j:j+1,:])
        #note_mono=np.argmax(pred_mono[i:i+1,j:j+1,:])
        hsl_proposed[i:i+1,j:j+1]=note_proposed
        #hsl_deepwavenet[i:i+1,j:j+1]=note_deepwavenet
        #hsl_tcn[i:i+1,j:j+1]=note_tcn
        #hsl_mono[i:i+1,j:j+1]=note_mono
        asli[i:i+1,j:j+1]=noteasli
        


        
asli_new=asli.flatten()
asli_new=asli_new.astype(int)
hsl_proposed_new=hsl_proposed.flatten()
hsl_proposed_b=hsl_proposed_new.astype(int)
hsl_deepwavenet_new=hsl_deepwavenet.flatten()
hsl_deepwavenet_new=hsl_deepwavenet_new.astype(int)
hsl_tcn_new=hsl_tcn.flatten()
hsl_tcn_new=hsl_tcn_new.astype(int)
hsl_mono_new=hsl_mono.flatten()
hsl_mono_new=hsl_mono_new.astype(int)

#hitung error
e_proposed=0
e_deep=0
e_tcn=0
e_mono=0
numsample=np.size(asli_new[0:50000],0)
for i in range(numsample):
    if (asli_new[i]!=hsl_proposed_new[i]).any():
        e_proposed=e_proposed+1
    if (asli_new[i]!=hsl_deepwavenet_new[i]).any():
        e_deep=e_deep+1
    if (asli_new[i]!=hsl_tcn_new[i]).any():
        e_tcn=e_tcn+1
    if (asli_new[i]!=hsl_mono_new[i]).any():
        e_mono=e_mono+1
        
 
durasi=np.size(asli_new,0)

np.savetxt('../Yasli.csv',asli_new,delimiter=',', fmt='%s')    
np.savetxt('../Yproposed.csv',hsl_proposed_new,delimiter=',', fmt='%s')   
np.savetxt('../Ydeepwavenet.csv',hsl_deepwavenet_new,delimiter=',', fmt='%s')  
np.savetxt('../Ytcn.csv',hsl_tcn_new,delimiter=',', fmt='%s') 
np.savetxt('../YmonoIRawNet.csv',hsl_mono_new,delimiter=',', fmt='%s')     
#check
import numpy as np
asli_new = np.loadtxt("Yasli.csv",
                 delimiter=",", dtype=str)
hsl_proposed_new=np.loadtxt("Yproposed.csv",
                 delimiter=",", dtype=str)
hsl_deepwavenet_new=np.loadtxt("Ydeepwavenet.csv",
                 delimiter=",", dtype=str)
hsl_tcn_new=np.loadtxt("Ytcn.csv",
                 delimiter=",", dtype=str)

durasi=50000
import matplotlib.pyplot as plt
#category=np.array([0,1,2,3,4,5,6,7])
x=np.arange(0,50000,1)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 20})
fig, axs = plt.subplots(5,figsize=(16, 23))

axs[0].plot(x,asli_new[0:durasi], marker='o',linestyle = 'None', color='black')

axs[0].xaxis.set_label_text('Sample')
axs[0].yaxis.set_label_text('Categorical Notes')
axs[0].set_title('(a) Target')
axs[0].set_xlim([0, durasi])
axs[0].set_ylim([0, 7])

axs[1].plot(x,hsl_tcn_new[0:durasi], marker='o',linestyle = 'None', color='green')
axs[1].set_xlim([0, durasi])
axs[1].set_ylim([0, 7])
axs[1].xaxis.set_label_text('Sample')
axs[1].yaxis.set_label_text('Categorical Notes')
axs[1].set_title('(b) Note transcription of TCN, Error = 19266 sample')


axs[2].plot(x,hsl_deepwavenet_new[0:durasi], marker='o',linestyle = 'None', color='purple')
axs[2].set_xlim([0, durasi])
axs[2].set_ylim([0, 7])
axs[2].xaxis.set_label_text('Sample')
axs[2].yaxis.set_label_text('Categorical Notes')
axs[2].set_title('(c) Note transcription of Deep WaveNet, Error = 5131 sample')


axs[3].plot(x,hsl_mono_new[0:durasi], marker='o',linestyle = 'None', color='red')
axs[3].set_xlim([0, durasi])
axs[3].set_ylim([0, 7])
axs[3].xaxis.set_label_text('Sample')
axs[3].yaxis.set_label_text('Categorical Notes')
axs[3].set_title('d) Note transcription of the monochannel IRawNet, Error = 10488 sample')

axs[4].plot(x,hsl_proposed_new[0:durasi], marker='o',linestyle = 'None', color='blue')
axs[4].set_xlim([0, durasi])
axs[4].set_ylim([0, 7])
axs[4].xaxis.set_label_text('Sample')
axs[4].yaxis.set_label_text('Categorical Notes')
axs[4].set_title('(e) Note transcription of the proposed method, Error = 4937 sample')


fig.tight_layout()
plt.savefig("resultrans_16.eps", dpi=1000)
plt.show()



