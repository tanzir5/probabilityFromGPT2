from transformers import GPT2Tokenizer, GPT2LMHeadModel

import numpy as np
import pandas as pd
import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print("DEVICE",DEVICE)

def get_tokenizer_model():
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
  tokenizer.pad_token = tokenizer.eos_token
  return (
    tokenizer,
    GPT2LMHeadModel.from_pretrained('gpt2-large').to(DEVICE),
  )

def get_texts(permute=False):
  def count_words(text):
    return len(text.split())
  df = pd.read_csv('aligned.csv')
  #df = df[:5]
  mask_column1 = df['prompt'].apply(count_words) >= 10
  mask_column2 = df['gen_text'].apply(count_words) >= 3
  df = df[mask_column1 & mask_column2]
  
  if not permute:
    return (df['prompt'].tolist(), df['gen_text'].tolist())
  else:
    return (df['prompt'].tolist(), np.random.permutation(df['gen_text'].tolist()))


  # return (
  #   [
  #     "rewrite the text in a different style: To Oliver's horror, the Dodger plunged his hand into the gentleman's pocket, drew out a handkerchief, and handed it to Bates.",
  #     "rewrite the text in a different style: You can imagine Oliver's horror when he saw him thrust his hand into the old gentleman's pocket, draw out a silk handkerchief and run off at full speed.",
  #     'rewrite the text in a different style: The cry of "Stop thief!" was raised',
  #     'rewrite the text in a different style: The cry of "Stop thief!" was raised',
  #     'rewrite the text in a different style: The cry of "Stop thief!" was raised',
  #     'rewrite the text in a different style: The cry of "Stop thief!" was raised',
  #   ],
  #   [
  #     "You can imagine Oliver's horror when he saw him thrust his hand into the old gentleman's pocket, draw out a silk handkerchief and run off at full speed.",
  #     "To Oliver's horror, the Dodger plunged his hand into the gentleman's pocket, drew out a handkerchief, and handed it to Bates.",
  #     'shouted "Stop thief!"',
  #     "they talked for hours",
  #     "Under the starry night sky, a gentle breeze rustled the leaves of the ancient oak tree, carrying with it the scent of blooming wildflowers, while distant laughter and the faint strumming of a guitar added a touch of magic to the tranquil scene, creating a moment of serenity that seemed to suspend time itself.",
  #     "The aroma of freshly baked bread filled the cozy kitchen, bringing a sense of warmth and comfort to the air."
  #   ],
  # )
  # [-2.3517279863633376, -2.3176996175407396, -3.908440732538476, -6.383984451858925, -2.694741470053277, -3.4071747528056093]



def get_gen_start_idx(prefixes, tokenizer):
  encoded_inputs = tokenizer(prefixes, return_tensors='pt', padding=True)
  return [
    torch.sum(attention_mask).item()
    for attention_mask in encoded_inputs['attention_mask']
  ]


def get_probability(prefixes, gen_texts, tokenizer, model):
  texts = [
    prefix + "\n" + " ".join(gen_text.split()[3:]) for prefix, gen_text in zip(prefixes, gen_texts)
  ]
  encoded_inputs = tokenizer(texts, return_tensors='pt', padding=True).to(DEVICE)
  # Create DataLoader for batching
  batch_size = 2
  dataset = torch.utils.data.TensorDataset(encoded_inputs['input_ids'], encoded_inputs['attention_mask'])
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
  scores = []
  with torch.no_grad():
      for batch in dataloader:
          input_ids, attention_mask = batch
          input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
          tmp_scores = model(input_ids=input_ids, attention_mask=attention_mask)['logits']
          scores.append(tmp_scores)

  # Concatenate the scores from all batches
  scores = torch.cat(scores, dim=0)
  #scores = model(**encoded_inputs)['logits']
  lengths = [
    torch.sum(attention_mask).item() 
    for attention_mask in encoded_inputs['attention_mask']
  ]
  gen_start_indices = get_gen_start_idx(prefixes, tokenizer)
  probabilities = []
  for batch_idx, score in tqdm(enumerate(scores)):
    # score is a nxk tensor
    #print("batch_idx", batch_idx)
    #print("score", score.shape)
    label = encoded_inputs['input_ids'][batch_idx]
    #print("label", label.shape)
    n = lengths[batch_idx]
    #print("n", n)
    c = np.array(
      [
        score[i][label[i+1]].item() 
        for i in range(gen_start_indices[batch_idx]-1, n-1)
      ]
    )
    #print("c", c.shape, c[0])
    log_exp_sum = np.log(
      [
        torch.sum(torch.exp(score[i])).item() 
        for i in range(gen_start_indices[batch_idx], n)
      ]
    )
    #print("log_exp_sum", log_exp_sum.shape, log_exp_sum[0])
    gen_length = n - gen_start_indices[batch_idx]
    #print(gen_length, len(c), len(log_exp_sum))
    assert(gen_length == len(c) == len(log_exp_sum))
    #print("c:",c)
    #print("sum of log_exp_sum", np.sum(log_exp_sum))
    y = np.sum(
      [
        c[i] - log_exp_sum[i]
        for i in range(gen_length)
      ]
    )
    y /= gen_length 
    #print("y", y)
    probabilities.append(y)
  return probabilities
    
if __name__ == '__main__':
  prefixes, gen_texts = get_texts(permute=False)
  prefixes_random, gen_texts_random = get_texts(permute=True)
  tokenizer, model = get_tokenizer_model()
  model.eval()
  real_probabilities = get_probability(prefixes, gen_texts, tokenizer, model)
  random_probabilities = get_probability(prefixes_random, gen_texts_random, tokenizer, model)
  with open('probability.json', 'w') as file:
    json.dump(
      {
        'real':real_probabilities,
        'random':random_probabilities,
      }, 
      file
    )