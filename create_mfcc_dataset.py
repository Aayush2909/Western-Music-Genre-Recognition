import librosa as lb
import os
import json
import math
import progressbar as pb
from time import sleep


DATASET  = "C:\\Genre Dataset"    #locaton of the sample music dataset
JSON = "C:\\mfcc_data.json"    #locaton of the json file to store data

TEST_DATA = "C:\\Test Data"    #locaton of the small size musc dataset for testing

SAMPLE_RATE = 22050
DUR = 30    #Duration in sec
SAMPLES = SAMPLE_RATE*DUR


#func to create mfcc data of music files
def create_mfcc_data(dataset_path, json_filepath, n_mfcc=13, n_fft=2048, hop_length=512, segments=5):
    
    
    #mfcc data with genre labels
    data = {
        "mapping" : [],
        "mfcc" : [],
        "label" : []
    }
    
    segment_size = SAMPLES//segments
    mfcc_vectors = math.ceil(segment_size/hop_length)
    
    
    #traverse through each and every files/folders of music dataset folder
    for i,(dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        if dirpath is not dataset_path:
            
            genre_path = dirpath.split('\\')
            genre = genre_path[-1]
            data["mapping"].append(genre)
            
            print("Processing {} ({}/10)-".format(genre, i))
            
            
            #progressbar initialisation
            widgets=[' [', progressbar.Timer(), '] ',pb.Bar('#', '[', ']'), ' ', pb.Percentage()]
            bar = pb.ProgressBar(maxval=100, widgets=widgets)
            sleep(0.25)
            bar.start()
            count = 1
            
            
            #processing each music file of a particular genre
            for f in filenames:
                
                filepath = os.path.join(dirpath, f)
                signal, sr = lb.load(filepath, sr=SAMPLE_RATE)    #loading signal of the music file
                
                bar.update(count)
                count+=1
                
                for s in range(segments):
                    
                    start = s*segment_size     #start of a sample of the respective signal
                    end = start + segment_size    #end of a sample of the respective signal
                    
                    mfcc = lb.feature.mfcc(signal[start:end], sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)    #calculating mfccs
                    mfcc = mfcc.T
                    
                    #store the mfcc
                    if len(mfcc) == mfcc_vectors:
                        data["mfcc"].append(mfcc.tolist())
                        data["label"].append(i-1)
                    
            bar.finish()
     
    
    #load the mfcc datas into a json file                    
    with open(json_filepath, 'w') as f:
        json.dump(data, f, indent=4)
        
    print("All Genres Processed...Done!!!")
