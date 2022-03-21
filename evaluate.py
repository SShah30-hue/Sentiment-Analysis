#libraries for transformers (text)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from transformers.models.auto.processing_auto import AutoProcessor
from modelling.text_modelling import BertForSentimentClassification, AlbertForSentimentClassification, DistilBertForSentimentClassification, RobertaForSentimentClassification
from dataset import TextDataset
from arguments import args
from torchmetrics.functional import accuracy
from sklearn.metrics import classification_report

#libraries for 2D CNN (audio)
import os
import pandas as pd
import numpy as np
from dataset import AudioDataset
from datasets import Dataset
from sklearn.model_selection import train_test_split
from dataset import prepare_data
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from tensorflow import keras 

#libraries for wav2vec2 (audio)
import torchaudio
import librosa
from transformers import  Wav2Vec2Processor
from modelling.audio_modelling import Wav2Vec2ForSpeechClassification

'''For evaluating transformers model (text) '''

def classification(model, dataloader, device):
    model.eval()
    encode_map = {'negative': 0,'neutral': 1,'positive': 2}
    reverse_encode_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    predictions = []
    label = []
    fileID = []
    

    with torch.no_grad():
        for input_ids, attention_mask, labels, FileID in tqdm(dataloader):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask)

            preds = torch.argmax(outputs,dim=1).squeeze().detach().cpu().tolist()
            labels = labels.squeeze().detach().cpu().tolist()
            FileID = str(list(FileID))[2: -2]
            predictions.append(preds)
            label.append(labels)
            fileID.append(FileID)


    #saving predictions 
    save_preds = pd.DataFrame({'fileID' : fileID, 'predictions' : predictions, 'label' : label})
    save_preds['predictions'].replace(reverse_encode_map, inplace=True)
    save_preds['label'].replace(reverse_encode_map, inplace=True)
    save_preds.to_csv('data/predictions/predictions.csv', index=False)
    #running classification report function
    cls_report = classification_report(predictions, label, target_names=encode_map)
    return cls_report 

def get_accuracy_from_outputs(outputs, labels):
    
    labels = torch.unsqueeze(labels, 1).int()
    pred = torch.argmax(outputs, dim=1)     
    acc = accuracy(pred, labels, num_classes=3)
    return acc

def evaluate(model, criterion, dataloader, device):
    model.eval()
    mean_acc, mean_loss, count = 0, 0, 0
    with torch.no_grad():
        for input_ids, attention_mask, labels, FileID in tqdm(dataloader, desc="Evaluating"):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask)
            mean_loss += criterion(input=outputs, target=labels)
            mean_acc += get_accuracy_from_outputs(outputs, labels)
            count += 1
            
    return mean_acc / count, mean_loss / count

'''For evaluating 2D CNN trained model (audio)'''

class get_results:
    
    def __init__(self, model, X_test, y_test, labels):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test             
        self.labels = labels

    def create_results(self, model):
        '''predict on test set and get accuracy results'''
        opt = Adam(0.001) #optimizer= "adam"
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        score = model.evaluate(X_test, y_test, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    def classification_report(self, X_test, y_test, model):

        encode_map = {'negative': 0,'neutral': 1,'positive': 2}

        preds = model.predict(X_test)
        preds= preds.argmax(axis=1)
        actual = y_test.argmax(axis=1)
        cls_report = classification_report(preds, actual, target_names=encode_map)
        print(cls_report)

'''For evaluating wav2vec2 model (audio) '''

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, target_sampling_rate)

    batch["speech"] = speech_array
    return batch


def predict(batch):
    features = processor(batch["speech"], sampling_rate=target_sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)
    #attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values).logits 

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    batch["predicted"] = pred_ids
    return batch

if __name__ == "__main__":

    if args.data_format == 'text':
        
        if args.model_name_or_path is None:
            args.model_name_or_path = 'roberta-base'

        #Configuration for the desired transformer model
        config = AutoConfig.from_pretrained(args.model_name_or_path)

        #Tokenizer for the desired transformer model
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        
        #Create the model with the desired transformer model
        if config.model_type == 'bert':
            model = BertForSentimentClassification.from_pretrained(args.model_name_or_path)
        elif config.model_type == 'albert':
            model = AlbertForSentimentClassification.from_pretrained(args.model_name_or_path)
        elif config.model_type == 'distilbert':
            model = DistilBertForSentimentClassification.from_pretrained(args.model_name_or_path)
        elif config.model_type == 'roberta':
            model = RobertaForSentimentClassification.from_pretrained(args.model_name_or_path)
        else:
            raise ValueError('This transformer model is not supported yet.')

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        #Takes as the input the logits of the positive class and computes the binary cross-entropy 
        criterion = nn.CrossEntropyLoss()

        val_set = TextDataset(filename='data/dev.tsv', maxlen=args.maxlen_val, tokenizer=tokenizer)
        val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, num_workers=args.num_threads)
        val_loader2 = DataLoader(dataset=val_set, batch_size=1, num_workers=args.num_threads)
        
        val_acc, val_loss = evaluate(model=model, criterion=criterion, dataloader=val_loader, device=device)
        print("Validation Accuracy : {}, Validation Loss : {}".format(val_acc, val_loss))

        cls_report = classification(model=model, dataloader=val_loader2, device=device)
        print(cls_report)

    elif args.data_format == 'audio_2dcnn':
        
        train_set = AudioDataset(filename='data/text/train_meld.tsv',filepath='data/audio/audio_meld/train', extension='wav')
        dev_set = AudioDataset(filename='data/text/dev_meld.tsv',filepath='data/audio/audio_meld/dev', extension='wav')
        
        #combining train and test as it feds into the model as one dataframe - train/test will occur automatically. 
        df1 = train_set.__addpath__()
        df2 = dev_set.__addpath__()
        dfs = [df1,df2]
        data = pd.concat(dfs, ignore_index = True)
        data = data
        
        if args.feature == 'mfcc':
            #mfcc extraction as denoted by "mfcc=1"
            mfcc = prepare_data(df=data, n=args.n_mfcc, mfcc=1, array_cols=args.array_cols, sampling_rate=args.sampling_rate, 
            audio_duration=args.audio_duration, n_mfcc=args.n_mfcc, n_melspec=args.n_melspec)
            #split between train and test 
            X_train, X_test, y_train, y_test = train_test_split(mfcc, data.Sentiment, train_size=args.train_size,test_size=args.test_size, shuffle=False)
            #one hot encode the target 
            lb = LabelEncoder()
            y_train = np_utils.to_categorical(lb.fit_transform(y_train))
            y_test = np_utils.to_categorical(lb.fit_transform(y_test))
            #normalization as per the standard NN process
            mean = np.mean(X_train, axis=0)
            std = np.std(X_train, axis=0)
            X_train = (X_train - mean)/std
            X_test = (X_test - mean)/std

        elif args.feature == 'melspec':
            #melspec extraction as denoted by "mfcc=0"
            specgram = prepare_data(df=data, n=args.n_melspec, mfcc=0, array_cols=args.array_cols, sampling_rate=args.sampling_rate, 
            audio_duration=args.audio_duration, n_mfcc=args.n_mfcc, n_melspec=args.n_melspec)
            #split between train and test 
            X_train, X_test, y_train, y_test = train_test_split(specgram, data.Sentiment, train_size=args.train_size,test_size=args.test_size, shuffle=False)
            #one hot encode the target 
            lb = LabelEncoder()
            y_train = np_utils.to_categorical(lb.fit_transform(y_train))
            y_test = np_utils.to_categorical(lb.fit_transform(y_test))
            #normalization as per the standard NN process
            mean = np.mean(X_train, axis=0)
            std = np.std(X_train, axis=0)
            X_train = (X_train - mean)/std
            X_test = (X_test - mean)/std
        
        #evaluate the trained model
        save_model_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'models', 'my_model_audio', '2dcnn'))
        saved_model = keras.models.load_model(save_model_path)
        results = get_results(saved_model, X_test, y_test, data.Sentiment.unique())
        results.create_results(saved_model)
        results.classification_report(X_test, y_test, saved_model)
    
    elif args.data_format == 'wav2vec2':
        
        dev_set = AudioDataset(filename='data/text/dev_meld.tsv', filepath='data/audio/audio_meld/dev', extension='wav')
        dev_set = dev_set.__addpath__()
        encode_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        dev_set['Sentiment'].replace(encode_map, inplace=True)

        dev_set = Dataset.from_pandas(dev_set[:args.test_size])

        print(dev_set)

        model_name_or_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'models', 'my_model_audio', 'wav2vec2', args.audio_model))
        config = AutoConfig.from_pretrained(model_name_or_path)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        target_sampling_rate = processor.feature_extractor.sampling_rate
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)

        dev_set = dev_set.map(speech_file_to_array_fn)
        result = dev_set.map(predict, batched=True, batch_size=args.per_device_eval_batch_size)

        label_names = [config.id2label[i] for i in range(config.num_labels)]

        y_true = [config.label2id[name] for name in result["Sentiment"]]
        y_pred = result["predicted"] 

        print(classification_report(y_true, y_pred, target_names=label_names))








