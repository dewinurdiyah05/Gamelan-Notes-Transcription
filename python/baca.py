# -*- coding: utf-8 -*-
"""
Created on Sat May 28 13:27:06 2022

@author: user
"""
#from skmultilearn.model_selection import iterative_train_test_split

import pandas as pd
import numpy as np
import librosa as lb
from foo import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import glob
import os.path
#import torch
#from torch_audiomentations import Compose, Gain, PolarityInversion
import soundfile as sf
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
def bacasignal(path):
     
    total=os.listdir(path)
    wavs=[]
    data=[]
    waktu=[]
    chanel1=[]
    chanel2=[]
    g=[]
    aug=[]
    k=1
    for filename in sorted(glob.glob(os.path.join(path, '*.wav'))):
        print(filename)
        data,sr = lb.load(filename,mono=False, sr=11024)
        
        
        #rescaling range 1 dan 0
        data1=-1+((data[0][0:]-np.min(data[0]))*(1-(-1)))/(np.max(data[0])-np.min(data[0]))
        data2=-1+((data[1][0:]-np.min(data[1]))*(1-(-1)))/(np.max(data[1])-np.min(data[1]))
        #data2=-1+((data[:,1:2]-np.min(data[:,1:2]))*(1-(-1)))/(np.max(data[:,1:2])-np.min(data[:,1:2]))
        
        #pakai data 30 detik pertama
        #data75detik=sr*75
        ch1=data1
        ch2=data2
        # #data=np.vstack((dat1, dat2))
        # #hsl=augmentasi(data,sr)
        # #sf.write("augmen/asli.wav", sr,data)
       
        # l=len(data)
        # grup = [k for i in range(l)]
        # grup=np.array(grup)
        duration1=np.floor(len(ch1)/sr)
       
        duration2=(len(ch1)/sr)-duration1
        duration=duration1+duration2
        #print(duration1, "-", duration2, "-", duration)
        time=np.arange(0,duration,1/sr)
        # #belakang koma 4 angka
        t=np.floor(time*1e3)/1e3
       
        chanel1.append(ch1)
        chanel2.append(ch2)
        waktu.append(t)
        # g.append(grup)
        # #aug.append(hsl)
        # k=k+1
        
    return  total,wavs,waktu,sr,g,data, chanel1, chanel2

  

def bacatarget(path):
    total=os.listdir(path)
    csvnotasi=[]
    waktutarget=[]
    number=[]
    grup=[]

    for filename in sorted(glob.glob(os.path.join(path, '*.csv'))):
        print("csv=",filename)
        df = np.array(pd.read_csv(filename,header=None))
        df1 = pd.DataFrame(df)
        target=df1.values[:,0:1]
        timetarget=np.floor(target*1e3)/1e3
        p=len(target)
        Xnumber=np.arange(1,p+1,1)
        number.append(Xnumber)
        #number=np.hstack((number,Xnumber))
        #number=number[1:]
        # timetarget1=np.reshape(timetarget, p)
        # Ywaktu=np.hstack((number,timetarget1))
        # Ywaktu=Ywaktu[1:]
        csvnotasi.append(df1.values)
        waktutarget.append(timetarget)
     
        
    return csvnotasi,waktutarget,Xnumber,p, number#,Ywaktu

def rescalenotasi(csvnotasi):
    
    for i in range(len(csvnotasi)):
        
        row=len(csvnotasi[i])
        for k in range(row):
            if (csvnotasi[i][k][1]==5).all():
                csvnotasi[i][k][1]=4
            if (csvnotasi[i][k][1]==6).all():
                csvnotasi[i][k][1]=5
            if (csvnotasi[i][k][1]==11).all():
                csvnotasi[i][k][1]=6
            if (csvnotasi[i][k][1]==60).all():
                csvnotasi[i][k][1]=7
                
                    
            
    
    return csvnotasi

    

def matchnotasi(chanel1, chanel2,t,ttarget,tsaron):
    
    noteinstrumen=[]
    amp1=[]
    amp2=[]
    #ampAugmentasi=[]
    for k in range(len(chanel1)):   
        tasli=np.array(t[k])
        #print(tasli.shape)
        #tasli=np.resize(tasli,[np.size(tasli,1)])
        ttar=np.array(ttarget[k])
        #ttar=np.resize(ttar,[np.size(ttar,1)])
        
        cc=np.array(tsaron[k])
        #cc=np.resize(cc,([np.size(cc,1),np.size(cc,2)]))
        cc=cc[:,1:2]
       
        index = [row for row in range(len(tasli))
                 for col in range(len(ttar)) if ttar[col] == tasli[row]]
            
        notasi=np.zeros(len(tasli))  
            
        i=0   
        for i in range(len(cc)-1):
                    #assign notasi
                    awal=index[i*11]
                    akhir=index[(i+1)*11]
                    #print(awal,"-",akhir)
                    notasi[awal:akhir]=cc[i] 
                    #print(awal,"-",akhir,"=",cc[i])
             
                
        
                
        notasi=notasi.astype(int)
        p=len(index)
        cut=index[p-1]
        #print(cut)
        wav1=chanel1[k][0:cut]
        wav2=chanel2[k][0:cut]
       
        notasi=notasi[0:cut]
        noteinstrumen.append(notasi)
        
        
        amp1.append(wav1)
        amp2.append(wav2)
       
    return index,amp1, amp2,noteinstrumen, wav1,wav2





def Bagiper10msdetik2(data1, data2,notasi,win,hop):
    #p=len(data1)
    bagi=win
    hop=hop
    j=0
    v2=[]
    v1=[]
    
    Y=[]
    z=0
    sample=np.size(data1)
        
    total=np.floor((sample/win))
    total=total.astype(int)
    print("total int =",total)
    #print('----------------',j)

    for i in range(total-1):
   
        awal=i*win
        akhir=(i+1)*win
        #awal=win*i
        #akhir=(i+1)*win
        print(awal,"-",akhir)
        v1.append(data1[awal:akhir])
        v2.append(data2[awal:akhir])
       
        Y.append(notasi[awal:akhir])
          
        
    vc1=np.array(v1)
    vc2=np.array(v2)
  
    
    #X1=np.resize(X,[np.size(X,0),np.size(X,1),1])
    #G1=np.resize(G,[np.size(G,0),np.size(G,1),1])
    #Xdata=np.concatenate((X1,G1),axis=2)
    Yasli=np.array(Y)
    # #encoder manual
    yhot=multi_class_vector(Yasli)
    
    return vc1,vc2,Y,Yasli,yhot, bagi,total,j,v1,v1


def multi_class_vector(Y):
    numrow=np.size(Y,0)
    numcol=np.size(Y,1)
    numclass=8
    Ytarget=np.zeros((numrow,numcol*numclass))
    #print("numrow:",numrow)
    #print("numcol:",numcol)
    #print("numclass:",numclass)
    for k in range(numrow):
        for u in range(numcol):
            #if (Y[k:k+1,u:u+1]==0).any():
            awal=u*numclass
            akhir=awal+numclass
            #print("k=",k,":",k+1,"u=",u,":",u+1,"awal=",awal,"-","akhir=",akhir)
            if (Y[k:k+1,u:u+1]==0).any():
                Ytarget[k:k+1,awal:akhir]=[1,0,0,0,0,0,0,0]
            if (Y[k:k+1,u:u+1]==1).any():
                Ytarget[k:k+1,awal:akhir]=[0,1,0,0,0,0,0,0]
            if (Y[k:k+1,u:u+1]==2).any():   
                Ytarget[k:k+1,awal:akhir]=[0,0,1,0,0,0,0,0]
            if (Y[k:k+1,u:u+1]==3).any():   
                Ytarget[k:k+1,awal:akhir]=[0,0,0,1,0,0,0,0]
            if (Y[k:k+1,u:u+1]==4).any():
                Ytarget[k:k+1,awal:akhir]=[0,0,0,0,1,0,0,0]
            if (Y[k:k+1,u:u+1]==5).any():
                Ytarget[k:k+1,awal:akhir]=[0,0,0,0,0,1,0,0]
            if (Y[k:k+1,u:u+1]==6).any():
                Ytarget[k:k+1,awal:akhir]=[0,0,0,0,0,0,1,0]
            if (Y[k:k+1,u:u+1]==7).any():
                Ytarget[k:k+1,awal:akhir]=[0,0,0,0,0,0,0,1]
    
       
    return Ytarget
def multi_class_raw(Y):
    numrow=np.size(Y,0)
    numcol=np.size(Y,1)
    numclass=8
    Ytarget=np.zeros((numrow,numcol*numclass))
    #print("numrow:",numrow)
    #print("numcol:",numcol)
    #print("numclass:",numclass)
    for k in range(numrow):
        for u in range(numcol):
            #if (Y[k:k+1,u:u+1]==0).any():
            awal=u*numclass
            akhir=awal+numclass
            #print("k=",k,":",k+1,"u=",u,":",u+1,"awal=",awal,"-","akhir=",akhir)
            if (Y[k:k+1,u:u+1]==0).any():
                Ytarget[k:k+1,awal:akhir]=[1,0,0,0,0,0,0,0]
            if (Y[k:k+1,u:u+1]==1).any():
                Ytarget[k:k+1,awal:akhir]=[0,1,0,0,0,0,0,0]
            if (Y[k:k+1,u:u+1]==2).any():   
                Ytarget[k:k+1,awal:akhir]=[0,0,1,0,0,0,0,0]
            if (Y[k:k+1,u:u+1]==3).any():   
                Ytarget[k:k+1,awal:akhir]=[0,0,0,1,0,0,0,0]
            if (Y[k:k+1,u:u+1]==4).any():
                Ytarget[k:k+1,awal:akhir]=[0,0,0,0,1,0,0,0]
            if (Y[k:k+1,u:u+1]==5).any():
                Ytarget[k:k+1,awal:akhir]=[0,0,0,0,0,1,0,0]
            if (Y[k:k+1,u:u+1]==6).any():
                Ytarget[k:k+1,awal:akhir]=[0,0,0,0,0,0,1,0]
            if (Y[k:k+1,u:u+1]==7).any():
                Ytarget[k:k+1,awal:akhir]=[0,0,0,0,0,0,0,1]
    
       
    return Ytarget





        
def reshapedata(Ytrain, Yvalid,win):
    #reshape -->(sample, timestep, feature)
    sampletrain=np.size(Ytrain,0)
    sampleval=np.size(Yvalid,0)
    timestep=win
    feature=1
   
    #reshape Y
    digit=np.size(Ytrain,1)
    featureY=int(digit/timestep)

    Yt=np.resize(Ytrain,[sampletrain,timestep,featureY])     
    Yv=np.resize(Yvalid,[sampleval,timestep,featureY])
    #Yst=np.resize(Ytest,[sampletest,timestep,featureY])
    return Yt,  Yv, timestep,feature,featureY





def splitdata1(X1,X2,Ytarget):
    totaldata=np.size(X1,0)
    inddata=np.arange(totaldata)
    
    Xtr1,Xts1,Ytr1, Yts1, indtrain1, indts1=train_test_split(X1,Ytarget,inddata, test_size=.2, random_state=42)
    #validasi dibagi dua. validasi 10% dan test 10%
    Xtr2,Xts2,Ytr2,Yts2,indvtrain2,indts2=train_test_split(X2,Ytarget, inddata, test_size=.2, random_state=42)
    
    

    return Xtr1,Xtr2,Ytr1, Ytr2,Xts1, Xts2,Yts1, Yts2


