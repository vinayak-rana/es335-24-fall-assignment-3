import torch.nn as nn

class NextWordModel(nn.Module):

    def __init__(self, vocab_size, emb_dim, context_len, hidden_dim,activation_fn):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.ll1 = nn.Linear(context_len*emb_dim, hidden_dim)
        
        self.ll2 = nn.Linear(hidden_dim,vocab_size)
        
        self.dropout = nn.Dropout(0.2)

        if activation_fn=='relu':
            self.actfn = nn.ReLU()

        if activation_fn=='tanh':
            self.actfn = nn.Tanh()
    
    def forward(self, x):

        e = self.embedding(x)
        
        flat_x = e.view(e.size(0),-1)
        h = self.ll1(flat_x)

        h = self.actfn(h)

        h = self.dropout(h)
        
        out = self.ll2(h)
        
        return out