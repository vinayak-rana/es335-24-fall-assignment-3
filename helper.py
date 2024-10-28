from model import NextWordModel
import torch.nn as nn
import torch
import json

def inference(embedding_len,context_len,num_lines,prompt,activation_fn):

    # loading the pretrained model
    model = NextWordModel(22582,embedding_len,context_len,1024,activation_fn)
    model.load_state_dict(torch.load(f'models/{embedding_len}_{context_len}_{activation_fn}.pth',map_location=torch.device('cpu')))
    
    # loading the vocab and mappings
    with open('mappings.json','r') as f:
        data = json.load(f)
    
    word_to_int = data['word_to_int']
    int_to_word = data['int_to_word']
    vocab = data['vocab']

    model.to(torch.device('cpu'))

    inp=[]
    final=''
    for w in prompt.split():
        if w in vocab:
            inp.append(word_to_int[w])
        else:
            inp.append(word_to_int['UNK_TKN'])

    if len(inp)<context_len:
        inp = [0] * (context_len - len(inp)) + inp
    else:
        inp = inp[-context_len:]

    inp = torch.tensor(inp)

    output = ""

    # inference step
    model.eval()
    with torch.no_grad():
        
        for _ in range(num_lines):
            while output!='.':

                op = model(inp.unsqueeze(dim=0))

                op = torch.tensor(torch.distributions.categorical.Categorical(logits = op).sample().item())

                inp = torch.cat((inp,op.unsqueeze(dim=0)),dim=0)
                inp=inp[1:]

                output = int_to_word[f'{int(op)}']

                final+=" "+output
            output = ''
    
    return final


# print(inference(64,8,2,'this is the best that he could do','relu'))