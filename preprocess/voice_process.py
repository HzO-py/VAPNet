import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.utils import make_chunks

path='/hdd/sdd/lzq/dianwang/dataset/voice_data/voice_data'
dataset_path='/hdd/sdc/hxj_dataset/电网数据'

problem=[]

def listwav(subpath):
    for subpath_file in os.listdir(subpath):
        ab_path=subpath+'/'+subpath_file
        if os.path.isdir(ab_path):
            listwav(ab_path)
        elif os.path.isfile(ab_path):
            file_name,file_ext=os.path.splitext(subpath_file)
            if file_ext=='.MP3':
                print(ab_path)
                try:
                    cutwav(ab_path)
                except:
                    problem.append(ab_path)

def cutwav(file):
    cut_size=3000
    audio = AudioSegment.from_file(file, "mp3")
    chunks=make_chunks(audio,cut_size)
    for i,chunk in enumerate(chunks):
        save_path=path+'/'+os.path.split(file)[-1]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cut_name=save_path+'/'+str(i)+'.wav'
        chunk.export(cut_name,format='wav')
        print(cut_name)

def wav_to_pic(path):
    for wav_files in os.listdir(path):
        try:
            save_path=path+'/'+wav_files+'/pic'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if len(os.listdir(path+'/'+wav_files))-1==len(os.listdir(save_path)):
                continue
            for wav in os.listdir(path+'/'+wav_files):
                wav_name,wav_ext=os.path.splitext(wav)
                if wav_ext!='.wav' and wav_ext!='.WAV':
                    continue
                y, sr = librosa.load(path+'/'+wav_files+'/'+wav, sr=None)
                melspec = librosa.feature.melspectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128)
                logmelspec = librosa.power_to_db(melspec)
                plt.figure()
                librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='mel')
                plt.axis('off')
                plt.savefig(save_path+'/'+wav_name+'.png',bbox_inches='tight',pad_inches=0.0)
            print(save_path)
        except:
            problem.append(wav_files)
            
wav_to_pic(path)
# listwav(dataset_path)
print(problem)