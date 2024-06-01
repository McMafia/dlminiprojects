import torch
import torch.nn as nn
import math

## torch.nn vs torch.functional.nn ->latter is stateless, we shud pass the parameters everytime, better flexibility

##multiplied embeddings by sqrt of d_model, check why

##unsqueeze operation -> adds an additional dimension wherever needed

##exp and log for numerical stability

##multi head attention allows parallelizability and as well helps the model to learn diverse features rather than generalizing
##splitting in multihead happens along the d_model dimension and not the seq len dimension (obvious!!!)

##each head has access to the entire sentence this way
##we can understand the op of each head as a low rank approximation if there were only one head

##Contiguous Memory: view() operates on contiguous memory, so it cannot change the order of elements. To achieve different memory layouts, use reshape() instead.

##nn embeddings takes in indices as input, might convert them into one hot vector or some representation before processing


class InputEmbeddings(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embeddings(x)*math.sqrt(self.d_model)

class PositionalEncodings(nn.Module):
    def __init__(self,d_model:int,seq_len:int,dropout:float)->None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len,d_model)
        #tensor of size (seq_len,1)
        position = torch.arange(0,seq_len,dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))

        #apply sin to even positions

        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0) #(1,seq_len,d_model)


        self.register_buffer('pe',pe)
    
    def forward(self,x):

        ## length of sentence might not be equal to the maximum sequence length 

        x = x + self.pe[:,:x.shape[1],:].requires_grad_(False)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):

    def __init__(self,eps:float = 1e-6)->None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) ##multiplied
        self.bias = nn.Parameter(torch.zeros(1)) ##added

    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True) ##(bs,seq_len,d_model) - >wo keepdim ->(bs,sl,1)
        std = x.std(dim=-1,keepdim=True)
        x = self.alpha*(x-mean)/(std+self.eps) + self.bias
        return x

class FeedForwardBlock(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropout:float)->None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        ##bs, seqlen,d_model
        ##l1->bs,sl,dff
        ##l2->bs,sl,dmodel
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,h:int,dropout:float)->None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model%h == 0,"d_model is not divisible"

        self.d_k = d_model//h

        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)

        self.w_o = nn.Linear(d_model,d_model)

        self.dropout = nn.Dropout()

    @staticmethod
    def attention(query, key,value, mask,dropout:nn.Dropout):
        d_k = query.shape[-1]
        
        attention_scores = (query@key.transpose(-2,-1))/math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
        
        attention_scores = attention_scores.softmax(dim=-1) #bs,h,sl,sl
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return attention_scores@value,attention_scores
    


    def forward(self,q,k,v,mask):
        query = self.w_q(q) ##bs,sl,dmodel->bs,sl,dmodel
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2) ##bs,h,sl,dk
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        
        x, attention_scores = MultiHeadAttention.attention(query,key,value,mask,self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k) #bs,h,sl,dk ->bs,sl,h,dk->bs,sl,dmodel

        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    def __init__(self,dropout:float)->None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttention,feed_forward_block:FeedForwardBlock,dropout:float)->None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(dropout)

        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self,x, src_mask):
        x = self.residual_connection[0](x, lambda x:self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connection[1](x,self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList)->None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttention,cross_attention_block:MultiHeadAttention,feed_forward_block:FeedForwardBlock,dropout:float)->None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x = self.residual_connections[0](x,lambda x : self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connections[1](x,lambda x : self.cross_attention_block(x, encoder_output,encoder_output,src_mask))
        x = self.residual_connections[2](x,self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList)->None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self,x,encoder_output,src_msk,tgt_msk):
        for layer in self.layers:
            x = layer(x,encoder_output,src_msk,tgt_msk)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self,d_model,vocab_size)->None:
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        #bs,sl,dmodel->bs,sl,vocabsz
        return torch.log_softmax(self.proj(x),dim=-1)
    
class Transformer(nn.Module):
    def __init__(self,encoder:Encoder,decoder:Decoder,src_embed:InputEmbeddings,tgt_embed:InputEmbeddings,src_pos:PositionalEncodings,tgt_pos:PositionalEncodings,projection_layer:ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos

        self.projection_layer = projection_layer
    
    def encode(self,src,src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self,encoder_output,src_mask,tgt,tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
    
    def project(self,x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size,tgt_vocab_size, src_seq_len, tgt_seq_len, d_model=512, N = 6, h = 8, dropout = 0.1, d_ff=2048)->Transformer:
    src_embed = InputEmbeddings(d_model,src_vocab_size)
    tgt_embed = InputEmbeddings(d_model,tgt_vocab_size)
    src_pos = PositionalEncodings(d_model,src_seq_len,dropout)
    tgt_pos = PositionalEncodings(d_model,tgt_seq_len,dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)
    
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model,h,dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model,tgt_vocab_size)

    transformer = Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)

    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    return transformer



    



        







        




