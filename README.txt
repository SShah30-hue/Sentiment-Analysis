Sentiment analysis neural network trained using RoBERTa, BERT, ALBERT, or DistilBERT, 2D CNN, and Wav2vec2 on MELD dataset.  

torch==1.3.0
pandas==0.25.0
numpy==1.17.4
transformers==3.0.1

To download data, please visit - https://affective-meld.github.io/
Note: To train or evaluate audio model, downloaded data needs to be converted from MP4 to WAV mono format using FFmpeg.
----------------------------------------------------------------------------------------------------------------------
For TEXT input:

TO TRAIN THE MODEL: 
python train.py --data_format text --model_name_or_path roberta-base --output_dir my_model --num_eps 2

TO EVALUATE THE MODEL YOU HAVE TRAINED:
python evaluate.py --data_format text --model_name_or_path models/my_model_text

TO ANALYZE THE INPUTS WITH THE MODEL YOU HAVE TRAINED
python analyze.py --model_name_or_path models/my_model_text

Sentiment analysis neural network trained by fine-tuning 2D CNN and Wav2vec 2.0 
on the MELD datasets.

----------------------------------------------------------------------------------------------------------------------
For AUDIO input:

2D CNN

TO TRAIN THE MODEL:
python train.py --data_format audio_2dcnn --feature mfcc --train_size 800 --test_size 200 --num_eps 20
OR
python train.py --data_format audio_2dcnn --feature mfcc --train_size 22000 --test_size 2131 --num_eps 20

{please note: normally the array_cols=641. If not, the error will display the right *array_cols* to input. Additionally,
you can choose between *mfcc* or *melspec* for feature extraction}

TO EVALUATE THE MODEL YOU HAVE TRAINED:
python evaluate.py --data_format audio_2dcnn --feature mfcc --train_size 800 --test_size 200

WAV2VEC2 

TO TRAIN THE MODEL:
python train.py --data_format wav2vec2 --train_size 800 --test_size 200

TO TRAIN THE WAV2VEC2 LARGE:
python train.py --data_format wav2vec2 --train_size 800 --test_size 200 --model_name_or_path facebook/wav2vec2-large-960h

TO EVALUATE THE MODEL YOU HAVE TRAINED:
python evaluate.py --data_format wav2vec2 --test_size 200 --audio_model checkpoint-xxx

