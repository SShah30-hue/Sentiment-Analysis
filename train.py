#libraries for transformers training (text)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AutoConfig, AutoTokenizer

from arguments import args
from dataset import TextDataset
from evaluate import evaluate
from modelling.text_modelling import (AlbertForSentimentClassification,
                                      BertForSentimentClassification,
                                      DistilBertForSentimentClassification,
                                      RobertaForSentimentClassification)

#libraries for 2D CNN training (audio) 
import os
import numpy as np
import pandas as pd
import pickle
from dataset import AudioDataset
from modelling.audio_modelling import get_2d_conv_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import tensorflow as tf
from dataset import prepare_data

#libraries for wav2vec2 training (audio)
from transformers import  Wav2Vec2Processor
import torchaudio
from transformers import EvalPrediction
from modelling.audio_modelling import Wav2Vec2ForSpeechClassification, DataCollatorCTCWithPadding, CTCTrainer
from transformers import TrainingArguments
from datasets import Dataset

'''For training transformers (text)'''

def train(model, criterion, optimizer, train_loader, val_loader, args):
    best_acc = 0
    for epoch in trange(args.num_eps, desc="Epoch"):
        model.train()
        for i, (input_ids, attention_mask, labels, FileID) in enumerate(tqdm(iterable=train_loader, desc="Training")):
            optimizer.zero_grad()  
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(input=outputs, target=labels) 
            loss.backward()
            optimizer.step()
        val_acc, val_loss = evaluate(model=model, criterion=criterion, dataloader=val_loader, device=device) #CHECK THIS
        print("Epoch {} complete! Validation Accuracy : {}, Validation Loss : {}".format(epoch, val_acc, val_loss))
        if val_acc > best_acc:
            print("Best validation accuracy improved from {} to {}, saving model...".format(best_acc, val_acc))
            best_acc = val_acc
            model.save_pretrained(save_directory=f'models/{args.output_dir}/')
            config.save_pretrained(save_directory=f'models/{args.output_dir}/')
            tokenizer.save_pretrained(save_directory=f'models/{args.output_dir}/')

'''preprocessing for wav2vec2 (audio)'''

def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def label_to_id(label, label_list):

    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label

def preprocess_function(examples):
    speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
    target_list = [label_to_id(label, label_list) for label in examples[output_column]]

    result = processor(speech_list, sampling_rate=target_sampling_rate)
    result["labels"] = list(target_list)

    return result

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

#remoing all audio files that are longer than 41 seconds 
def remove_long_common_voicedata(dataset, max_seconds=41):
    dftest= dataset.to_pandas()
    dftest['len']= dftest['input_values'].apply(len)
    maxLength = max_seconds*16000 
    dftest= dftest[dftest['len']<maxLength]
    dftest = dftest.drop('len', 1)
    dataset= dataset.from_pandas(dftest)
    del dftest
    return dataset
        

if __name__ == "__main__":

    #training text
    if args.data_format == 'text':
        
        if args.model_name_or_path is None:
            args.model_name_or_path = 'roberta-base'

    
        #Configuration for the desired transformer model
        config = AutoConfig.from_pretrained(args.model_name_or_path)

        #Tokenizer for the desired transformer model
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        #Create the model with the desired transformer model
        if config.model_type == 'bert':
            model = BertForSentimentClassification.from_pretrained(args.model_name_or_path, config=config)
        elif config.model_type == 'albert':
            model = AlbertForSentimentClassification.from_pretrained(args.model_name_or_path, config=config)
        elif config.model_type == 'distilbert':
            model = DistilBertForSentimentClassification.from_pretrained(args.model_name_or_path, config=config)
        elif config.model_type == 'roberta':
            model = RobertaForSentimentClassification.from_pretrained(args.model_name_or_path, config=config)
        else:
            raise ValueError('This transformer model is not supported yet.')

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        #Takes as the input the logits of the positive class and computes the cross-entropy 
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

        train_set = TextDataset(filename='data/text/train_meld.tsv', maxlen=args.maxlen_train, tokenizer=tokenizer)
        val_set = TextDataset(filename='data/text/dev_meld.tsv', maxlen=args.maxlen_val, tokenizer=tokenizer)

        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, num_workers=args.num_threads)
        val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, num_workers=args.num_threads)

        train(model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader, args=args)

    #training audio
    elif args.data_format == 'audio_2dcnn':
                
        train_set = AudioDataset(filename='data/text/train_meld.tsv',filepath='data/audio/audio_meld/train', extension='wav')
        train_set2 = AudioDataset(filename='data/text/train_crema.tsv',filepath='data/audio/audio_crema', extension='wav')
        train_set3 = AudioDataset(filename='data/text/train_ravdess.tsv',filepath='data/audio/audio_ravdess', extension='wav')
        train_set4 = AudioDataset(filename='data/text/train_emo.tsv',filepath='data/audio/audio_emo', extension='wav')
        dev_set = AudioDataset(filename='data/text/dev_meld.tsv',filepath='data/audio/audio_meld/dev', extension='wav')
        
        #combining train and test as it feds into the model as one dataframe - train/test will occur automatically. 
        train_set = train_set.__addpath__()
        train_set2 = train_set2.__addpath__()
        train_set3 = train_set3.__addpath__()
        train_set4 = train_set4.__addpath__()
        dev_set = dev_set.__addpath__()
        dfs = [train_set, train_set2, train_set3, train_set4, dev_set]
        #dfs = [train_set, dev_set]
        data = pd.concat(dfs, ignore_index = True)
        data = data
        print(data)
    

        if args.feature == 'mfcc':
            dirpath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'models', 'my_model_audio', '2dcnn'))
            #mfcc extraction as denoted by "mfcc=1"
            mfcc = prepare_data(df=data, n=args.n_mfcc, mfcc=1, array_cols=args.array_cols, sampling_rate=args.sampling_rate, 
            audio_duration=args.audio_duration, n_mfcc=args.n_mfcc, n_melspec=args.n_melspec)
            #split between train and test 
            X_train, X_test, y_train, y_test = train_test_split(mfcc, data.Sentiment, train_size=args.train_size,test_size=args.test_size, shuffle=True)
            #one hot encode the target 
            lb = LabelEncoder()
            y_train = np_utils.to_categorical(lb.fit_transform(y_train))
            y_test = np_utils.to_categorical(lb.fit_transform(y_test))
            #normalization as per the standard NN process
            mean = np.mean(X_train, axis=0)
            std = np.std(X_train, axis=0)
            X_train = (X_train - mean)/std
            X_test = (X_test - mean)/std
            #saving mean and std variables
            pickle.dump([mean, std], open(str(dirpath) + "/norm_vals.pkl", 'wb'))
            #run CNN model training
            with tf.device("/gpu:0"):
                model = get_2d_conv_model(n=args.n_mfcc, array_cols=args.array_cols)
                model_history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=args.batch_size, verbose = 2, epochs=args.num_eps)
                save_model_path = dirpath 
                #saving model to path
                model.save(save_model_path)
                print("Model saved to: ", save_model_path) 

        elif args.feature == 'melspec':
            dirpath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'models', 'my_model_audio', '2dcnn'))
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
            #saving mean and std variables
            pickle.dump([mean, std], open(str(dirpath) + "/norm_vals.pkl", 'wb'))
            #run CNN model training
            with tf.device("/gpu:0"):
                model = get_2d_conv_model(n=args.n_melspec, array_cols=args.array_cols)
                model_history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=args.batch_size, verbose = 2, epochs=args.num_eps)
                save_model_path = dirpath 
                #save model to path
                model.save(save_model_path)
                print("Model saved to: ", save_model_path)
                
        else:
            raise ValueError("Please input the argument 'feature' and try again")



    elif args.data_format == 'wav2vec2':
        
        train_set = AudioDataset(filename='data/text/train_meld.tsv',filepath='data/audio/audio_meld/train', extension='wav')
        train_set2 = AudioDataset(filename='data/text/train_crema.tsv',filepath='data/audio/audio_crema', extension='wav')
        train_set3 = AudioDataset(filename='data/text/train_ravdess.tsv',filepath='data/audio/audio_ravdess', extension='wav')
        train_set4 = AudioDataset(filename='data/text/train_emo.tsv',filepath='data/audio/audio_emo', extension='wav')
        dev_set = AudioDataset(filename='data/text/dev_meld.tsv',filepath='data/audio/audio_meld/dev', extension='wav')
        
        #combining train and test as it feds into the model as one dataframe - train/test will occur automatically.
        train_set = train_set.__addpath__()
        dev_set = dev_set.__addpath__()
        train_set2 = train_set2.__addpath__()
        train_set3 = train_set3.__addpath__()
        train_set4 = train_set4.__addpath__()
        dfs = [train_set, train_set2, train_set3, train_set4]

        train_set = pd.concat(dfs, ignore_index = True)
        train_set = train_set        
        #pd.set_option('display.max_colwidth', None)

        print(train_set)
        print(dev_set)

        encode_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        train_set['Sentiment'].replace(encode_map, inplace=True)
        dev_set['Sentiment'].replace(encode_map, inplace=True)

        train_set = Dataset.from_pandas(train_set[:args.train_size])
        dev_set = Dataset.from_pandas(dev_set[:args.test_size])

        #specifying the input and output column
        input_column = "path"
        output_column = "Sentiment"

        #distinguishing the unique labels in our SER dataset
        label_list = train_set.unique(output_column)
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)
        print(f"A classification problem with {num_labels} classes: {label_list}")

        #specifying model name, config and processor
        model_name_or_path = args.model_name_or_path
        if args.model_name_or_path is None:
            model_name_or_path = "facebook/wav2vec2-base-960h"
        pooling_mode = "mean"

        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            mask_time_prob=0.00,
            label2id={label: i for i, label in enumerate(label_list)},
            id2label={i: label for i, label in enumerate(label_list)},
            finetuning_task="wav2vec2_clf",
        )
        setattr(config, 'pooling_mode', pooling_mode)

        processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
        target_sampling_rate = processor.feature_extractor.sampling_rate

        train_set = train_set.map(preprocess_function,batch_size=100,batched=True,num_proc=1)
        dev_set = dev_set.map(preprocess_function,batch_size=100,batched=True,num_proc=1)
        
        train_set = remove_long_common_voicedata(train_set, max_seconds=20)
        dev_set = remove_long_common_voicedata(dev_set, max_seconds=20)

        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
        
        is_regression = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path,config=config).to(device)
        model.freeze_feature_extractor()


        training_args = TrainingArguments(
            output_dir=os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'my_model_audio', 'wav2vec2')),
            overwrite_output_dir=True,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=2,
            evaluation_strategy="steps",
            num_train_epochs=args.num_eps,
            gradient_checkpointing=True,
            save_steps=args.eval_steps,
            eval_steps=args.eval_steps,
            logging_steps=args.logging_steps,
            learning_rate=args.lr,
            weight_decay=0.005,
            warmup_steps=args.warmup_steps,
            save_total_limit=1,
            load_best_model_at_end= True

        )

        trainer = CTCTrainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_set,
            eval_dataset=dev_set,
            tokenizer=processor.feature_extractor,
        )

        trainer.train()

    
    else:
        raise ValueError("Please input the argument 'data_format' and try again")