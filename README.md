# Gamelan-Notes-Transcription
This repository provides data sources in the "data" directory, python code in the "python" directory, saved model in the "model_save" directory, and log training in the "log" directory.

===============Important expalanations====================
1. The "data" directory contains song recordings that are played with a single instrument of saron barung and saved in wav extension, 
2. Each raw audio has a target file in csv. The csv file consists of two columns, the first column is an onset time of a note played, and the second column is a note.
3. The "python" directory contains a main program that is saved by the name of "utama.py," two supported files of "baca.py" and "deeplearning_klasifikasi.py."
4. "baca.py" contains some functions to read raw audio and target, matching note target with raw audio time, slicing window, spliting data, and reshaping data.
5. "deeplearning_klasifikasi.py" provides IRawNet multi and mono channel; Deepwavenet; and TCN functions
4. Python version : 3.8
5. Libraries in anaconda environment: Tensorflow ver.2, numpy, imblearn, librosa, and matplotlib
