#libraries for text input
import pandas as pd
from torch.utils.data import Dataset
#libraries for audio input
import os
import glob
import natsort
import librosa
from tqdm import tqdm
import numpy as np


''' Following is the class for reading and processing transformers TEXT input'''

class TextDataset(Dataset):

    def __init__(self, filename, maxlen, tokenizer): 
        #Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter='\t')
        #Initialize the tokenizer for the desired transformer model
        self.tokenizer = tokenizer
        #Maximum length of the tokens list to keep all the sequences of fixed size
        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):

        #Select the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 'Utterance']
        label = self.df.loc[index, 'Sentiment']
        FileID = self.df.loc[index, 'FileID']    
    
        #Preprocess the text to be suitable for the transformer
        tokens = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.maxlen,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        ) 
        
        input_ids = tokens["input_ids"].flatten()
        attention_mask = tokens["attention_mask"].flatten()
        label = label
        FileID = FileID


        return input_ids, attention_mask, label, FileID


''' Following is the class for reading and processing 2D CNN AUDIO input'''

class AudioDataset(Dataset):

    def __init__(self, filename, filepath, extension):

        #Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter = '\t')
        #Define filepath
        self.filepath = filepath
        #Define extension
        self.extension = extension
        #Define parent path
        self.parent_dir = os.path.dirname(os.path.abspath(__file__))
        
    def __len__(self):
        return len(self.df)


    def __addpath__(self):

        os.chdir(os.path.join(self.parent_dir, self.filepath))
        result = glob.glob('*.{}'.format(self.extension))
        result = natsort.natsorted(result, reverse=False)
        path = []
        for i in result:
            path.append(os.path.join(self.parent_dir, self.filepath, i).replace("\\", "/"))
        
        data = self.df
        #add audio filename as a column
        #data['name'] = result
        #add audio filepath as a column
        data['path'] = path
        
        return data

def prepare_data(df, n, mfcc, array_cols, sampling_rate, audio_duration, n_mfcc, n_melspec):
    X = np.empty(shape=(df.shape[0], n, array_cols, 1))
    input_length = sampling_rate * audio_duration
    
    #loop feature extraction over the entire dataset
    cnt = 0
    for fname in tqdm(df.path):
        file_path = fname
        data, _ = librosa.load(file_path, sr=sampling_rate,res_type="kaiser_fast",duration=41,offset=0.0)

        #random offset/padding
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, int(input_length) - len(data) - offset), "constant")
        
        #which feature?
        if mfcc == 1:
            #MFCC extraction 
            MFCC = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=n_mfcc)
            MFCC = np.expand_dims(MFCC, axis=-1)
            X[cnt,] = MFCC
            
        else:
            #melspectogram
            melspec = librosa.feature.melspectrogram(data, n_mels = n_melspec)   
            logspec = librosa.amplitude_to_db(melspec)
            logspec = np.expand_dims(logspec, axis=-1)
            X[cnt,] = logspec
            
        cnt += 1
    
    return X







