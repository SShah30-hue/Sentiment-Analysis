from argparse import ArgumentParser

parser = ArgumentParser()

#arguments for both text and audio
parser.add_argument('--data_format', type=str, default=None, help='Which data format to train.')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size during training.')
parser.add_argument('--num_eps', type=int, default=2, help='Number of training epochs.')
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate for Adam.')
parser.add_argument(
    '--model_name_or_path', 
    type=str, 
    default=None, 
    help='''Name of or path to the pretrained/trained model. 
                    For training choose between bert-base-uncased, albert-base-v2, distilbert-base-uncased etc. 
                    For evaluating/analyzing/server choose between paths to the models you have trained previously.'''
)

#arguments for text 
parser.add_argument('--maxlen_train', type=int, default=60, help='Maximum number of tokens in the input sequence during training.')
parser.add_argument('--maxlen_val', type=int, default=60, help='Maximum number of tokens in the input sequence during evaluation.')
parser.add_argument('--num_threads', type=int, default=0, help='Number of threads for collecting the datasets.')
parser.add_argument('--output_dir', type=str, default='my_model_text', help='Where and with what name to save the trained model.')

#arguments for 2D CNN and Wav2vec2
parser.add_argument('--sampling_rate', type=int, default=8000, help='Sampling rate in Hz of an audio file.')
parser.add_argument('--audio_duration', type=int, default=41, help='Maximum number of audio duration.')
parser.add_argument('--n_mfcc', type=int, default=30, help='Number of MFCCs across an audio signal to compute.')
parser.add_argument('--n_melspec', type=int, default=60, help='Number of MFCCs across an audio signal to compute.')
parser.add_argument('--feature', type=str, default=None, help='Name of the feature to be extracted. Please choose between mfcc or melspec')
parser.add_argument('--array_cols', type=int, default=641, help='Shape of an array of input dimensions.')
parser.add_argument('--train_size', type=int, default=9988, help='Number of training.')
parser.add_argument('--test_size', type=int, default=1108, help='Number of testing.')

parser.add_argument('--audio_model', type=str, default=None, help='dir name where trained model best checkpoint is stored.')
parser.add_argument('--save_steps', type=int, default=500, help='Save checkpoint every X updates steps.')
parser.add_argument('--eval_steps', type=int, default=500, help='Run an evaluation every X steps.')
parser.add_argument('--logging_steps', type=int, default=500, help='Log every X updates steps.')
parser.add_argument('--warmup_steps', type=int, default=1000, help='Linear warmup over warmup_steps.')
parser.add_argument('--per_device_train_batch_size', type=int, default=2, help='Batch size per GPU/TPU core/CPU for training.')
parser.add_argument('--per_device_eval_batch_size', type=int, default=2, help='Batch size per GPU/TPU core/CPU for evaluating.')



args = parser.parse_args()