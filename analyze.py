import torch
from transformers import AutoTokenizer, AutoConfig
from modelling.text_modelling import BertForSentimentClassification, AlbertForSentimentClassification, DistilBertForSentimentClassification, RobertaForSentimentClassification
from arguments import args

def classify_sentiment(sentence, tokenizer, model):
    with torch.no_grad():
        tokens = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=maxlen,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors='pt',
        )

        outputs = model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
        prob = outputs.flatten().numpy()
        prob = prob*100
        
        encode_map = {'negative': 0,'neutral': 1,'positive': 2}
        
        for label, prob in zip(encode_map, prob):
             print(f"{label} with probability of: {prob}%")
            
if __name__ == "__main__":

    if args.model_name_or_path is None:
        args.model_name_or_path = 'roberta-base'

    #Configuration for the desired transformer model
    config = AutoConfig.from_pretrained(args.model_name_or_path)

    print('Please wait while the analyser is being prepared.')

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

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = model.to(device)
    
    model.eval()

    #Initialize the tokenizer for the desired transformer model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    maxlen = args.maxlen_val
    
    sentence = input('Input sentiment to analyze: ')
    while sentence:
        classify_sentiment(sentence, tokenizer=tokenizer, model=model)
        sentence = input('Input sentiment to analyze: ')