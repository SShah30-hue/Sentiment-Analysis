#libraries for text input
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, AlbertPreTrainedModel, AlbertModel, DistilBertPreTrainedModel, DistilBertModel, RobertaPreTrainedModel, RobertaModel

'''Models for text input'''

class RobertaForSentimentClassification(RobertaPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        for parameter in self.roberta.parameters():
            parameter.require_grad = False
        #The classification layer that takes the [CLS] representation and outputs the logit
        self.classifier = nn.Linear(config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
     
        #Feed the input to Roberta model to obtain outputs
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_reps = self.classifier(outputs.pooler_output)
        logits = torch.softmax(cls_reps, dim=1)
        return logits
    
class BertForSentimentClassification(BertPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        for parameter in self.bert.parameters():
            parameter.require_grad = False
        #The classification layer that takes the [CLS] representation and outputs the logit
        self.classifier = nn.Linear(config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
    
        #Feed the input to Bert model to obtain outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_reps = self.classifier(outputs.pooler_output)
        logits = torch.softmax(cls_reps, dim=1)
        return logits
    
class AlbertForSentimentClassification(AlbertPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.albert = AlbertModel(config)
        for parameter in self.albert.parameters():
            parameter.require_grad = False
        #The classification layer that takes the [CLS] representation and outputs the logit
        self.classifier = nn.Linear(config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
      
        #Feed the input to Albert model to obtain outputs
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask)
        cls_reps = self.classifier(outputs.pooler_output)
        logits = torch.softmax(cls_reps, dim=1)
        return logits

class DistilBertForSentimentClassification(DistilBertPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        for parameter in self.distilbert.parameters():
            parameter.require_grad = False
        #The classification layer that takes the [CLS] representation and outputs the logit
        self.classifier = nn.Linear(config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
       
        #Feed the input to Distilbert model to obtain outputs
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_reps = self.classifier(outputs.pooler_output)
        logits = torch.softmax(cls_reps, dim=1)
        return logits



