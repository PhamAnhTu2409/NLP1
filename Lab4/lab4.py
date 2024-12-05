### Практическая работа "Маскирование слов"
# voice 
# truth
# Preparation
from transformers import BertTokenizer, BertForMaskedLM     
# Bertokenizer used for converting text into numerical data
# BertForMaskedLM - a pretrained bert model for the task of filling in missing words
from torch.nn import functional as F        #used to convert logits into probabilities
import torch                                #used for working with tensors and performing computations on GPUs 

# Initialization
name = "bert-base-multilingual-uncased"
tokenizer = BertTokenizer.from_pretrained(name)
model = BertForMaskedLM.from_pretrained(name, return_dict = True)

# Computation
text = "No one wants to hear the" + tokenizer.mask_token + "if it is not what they want to hear."
input = tokenizer.encode_plus(text, return_tensors = "pt")
mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
output = model(**input)

# Output
logits = output.logits
softmax = F.softmax(logits, dim = -1)
mask_word = softmax[0, mask_index[0], :]
top = torch.topk(mask_word, 10)
for token in top[-1][0].data:
    print(tokenizer.decode([token]))

