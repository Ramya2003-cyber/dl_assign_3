"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple
import spacy
from spacy.cli import download
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION  
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    qkt=torch.matmul(Q,K.transpose(-2,-1)) 
    d_k=Q.size(-1)
    scaled_qkt=qkt/math.sqrt(d_k)
    if mask is not None:
        scaled_qkt=scaled_qkt.masked_fill(mask, float('-inf'))
    attn_w=F.softmax(scaled_qkt,dim=-1)
    out=torch.matmul(attn_w,V)
    return out, attn_w

# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS 
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    src_mask=(src == pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    tgt_pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, tgt_len]
    tgt_len = tgt.shape[1]
    tgt_sub_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=tgt.device), diagonal=1).bool()  # [tgt_len, tgt_len]
    tgt_mask = tgt_pad_mask | tgt_sub_mask  # Combine padding and look-ahead
    return tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION 
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head
        self.W_q=nn.Linear(d_model, d_model)
        self.W_k=nn.Linear(d_model, d_model)    
        self.W_v=nn.Linear(d_model, d_model)
        self.W_o=nn.Linear(d_model, d_model)
        self.dropout=nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]

        """
        q=self.W_q(query) #linear projections
        k=self.W_k(key)    
        v=self.W_v(value)
        batch_size=query.size(0)
        q=q.view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2) #split into heads and rearrange
        k=k.view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        v=v.view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        _, attn_w=scaled_dot_product_attention(q,k,v,mask)
        attn_w=self.dropout(attn_w) 
        attn_out=torch.matmul(attn_w,v) 
        attn_out=attn_out.transpose(1,2).contiguous().view(batch_size,-1,self.d_model) #rearrange and concatenate heads
        out=self.W_o(attn_out) 
        return out


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING  
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe=torch.zeros(max_len, d_model)
        position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]  

        """
        x=x+self.pe[:,:x.size(1),:]
        return self.dropout(x) 


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK 
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO: Task 2.3 — define:
        #   self.linear1 = nn.Linear(d_model, d_ff)
        #   self.linear2 = nn.Linear(d_ff, d_model)
        #   self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]

        """
        attn_out=self.self_attn(x,x,x,src_mask)
        x=self.norm1(x+self.dropout1(attn_out)) 
        ffn_out=self.ffn(x)
        x=self.norm2(x+self.dropout2(ffn_out)) 
        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER 
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn=PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)

        self.dropout1=nn.Dropout(dropout)
        self.dropout2=nn.Dropout(dropout)
        self.dropout3=nn.Dropout(dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        attn_out1=self.self_attn(x,x,x,tgt_mask)
        x=self.norm1(x+self.dropout1(attn_out1))
        attn_out2=self.cross_attn(x,memory,memory,src_mask)
        x=self.norm2(x+self.dropout2(attn_out2))
        ffn_out=self.ffn(x)
        x=self.norm3(x+self.dropout3(ffn_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers=nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        d_model=layer.norm1.normalized_shape[0]
        self.norm=nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers=nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        d_model=layer.norm1.normalized_shape[0]
        self.norm=nn.LayerNorm(d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,

    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """

    def __init__(
        self,
        src_vocab_size: int=7853,
        tgt_vocab_size: int=5893,
        d_model:   int   = 256,
        N:         int   = 3,
        num_heads: int   = 8,
        d_ff:      int   = 512,
        dropout:   float = 0.2,
        checkpoint_path: str = "best_model_weights.pt",
    ) -> None:
        super().__init__()
        try:
            spacy.load("de_core_news_sm")
        except OSError:
            download("de_core_news_sm")
            
        # Download English
        try:
            spacy.load("en_core_web_sm")
        except OSError:
            download("en_core_web_sm")
        try:
            from dataset import Multi30kDataset
            self.dataset_helper=Multi30kDataset(split='train')
            self.dataset_helper.build_vocab()
            self.de_vocab=self.dataset_helper.de_vocab
            self.en_vocab=self.dataset_helper.en_vocab
            self.spacy_de=self.dataset_helper.spacy_de
            self.spacy_en=self.dataset_helper.spacy_en
        except Exception as e:
            print("Error loading dataset helper inside Transformer:", e)
        
        self.src_embed=nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed=nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc=PositionalEncoding(d_model, dropout)
        enc_layer=EncoderLayer(d_model,num_heads,d_ff,dropout)
        self.encoder=Encoder(enc_layer,N)
        dec_layer=DecoderLayer(d_model,num_heads,d_ff,dropout)
        self.decoder=Decoder(dec_layer,N)
        self.generator=nn.Linear(d_model,tgt_vocab_size)

        self.model_config = {
            'src_vocab_size': src_vocab_size,
            'tgt_vocab_size': tgt_vocab_size,
            'd_model': d_model,
            'N': N,
            'num_heads': num_heads,
            'd_ff': d_ff,
            'dropout': dropout
        }
        
        # init should also load the model weights if checkpoint path provided, download the .pth file like this
        # 1. Autograder Bug Fix: Force the checkpoint path if the TA forgot to pass it
        if checkpoint_path is None:
            checkpoint_path = "best_model_weights.pt"
            
        # 2. Download the weights from Google Drive if they aren't on the server yet
        if not os.path.exists(checkpoint_path):
            import gdown
            drive_id = "1h53ElXsT-SmUr90IrxOGPeTbPaOGFQY1"  # <-- Make sure to paste your actual ID here!
            gdown.download(id=drive_id, output=checkpoint_path, quiet=True)
            
        # 3. Safely load the weights
        # 3. Safely load the weights (REPLACE YOUR OLD BLOCK WITH THIS)
        if os.path.exists(checkpoint_path):
            try:
                # Load to CPU safely
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                state_dict = checkpoint['model_state_dict']
                
                # Proactively fix the Kaggle DataParallel bug
                clean_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('module.'):
                        # Strip "module." from the string
                        clean_state_dict[key[7:]] = value
                    else:
                        clean_state_dict[key] = value
                
                # Load the cleaned weights into the model
                self.load_state_dict(clean_state_dict)
                print("WEIGHTS SUCCESSFULLY LOADED AND APPLIED!")
                
            except Exception as e:
                # If PyTorch still rejects the weights, PRINT THE REASON to the console!
                print(f"CRITICAL WEIGHT ERROR: {e}")
        
        '''if checkpoint_path is not None:
            # Prevent re-downloading every single time you initialize the model
            if not os.path.exists(checkpoint_path):
                print("Downloading model weights from Google Drive...")
                # TODO: Replace this ID with the Google Drive link to your trained .pt file!
                drive_id = "<YOUR_DRIVE_FILE_ID_HERE>" 
                gdown.download(id=drive_id, output=checkpoint_path, quiet=False)
            
            # Load the weights into the model
            if os.path.exists(checkpoint_path):
                print(f"Loading weights from {checkpoint_path}...")
                # map_location="cpu" prevents CUDA out-of-memory errors on the autograder
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                # Extract just the model state dict (ignoring optimizer/scheduler states)
                self.load_state_dict(checkpoint['model_state_dict'])'''

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
    
        sec_emb=self.src_embed(src) 
        src_emb=self.pos_enc(sec_emb)
        return self.encoder(src_emb,src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        tgt_emb=self.tgt_embed(tgt) 
        tgt_emb=self.pos_enc(tgt_emb)
        dec_out=self.decoder(tgt_emb,memory,src_mask,tgt_mask)
        return dec_out

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory=self.encode(src,src_mask)
        dec_out=self.decode(memory, src_mask, tgt, tgt_mask)
        logits=self.generator(dec_out)
        return logits


    def infer(self, src_sentence: str) -> str:
        """
        Translates a German sentence to English using greedy autoregressive decoding.
        
        Args:
            src_sentence: The raw German text.
            
            
        Returns:
            The fully translated English string, detokenized and clean.
        """
        self.eval()
        device=next(self.parameters()).device
        tokens = [token.text.lower() for token in self.spacy_de.tokenizer(src_sentence)]
        tokens=['<sos>']+tokens+['<eos>']
        unk_idx=self.de_vocab.stoi.get('<unk>', 0)
        src_indices=[self.de_vocab.stoi.get(tok, unk_idx) for tok in tokens]
        src_tensor=torch.LongTensor(src_indices).unsqueeze(0).to(device)
        src_mask=torch.ones((1,1,1,src_tensor.shape[1]),dtype=torch.bool).to(device)
        with torch.no_grad():
            memory=self.encode(src_tensor,src_mask)
        tgt_indices=[self.en_vocab.stoi['<sos>']]
        for i in range(50):
            trg_tensor=torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
            seq_len=trg_tensor.shape[1]
            trg_mask=torch.tril(torch.ones((1,1,seq_len,seq_len),dtype=torch.bool)).to(device)
            with torch.no_grad():
                out=self.decode(memory, src_mask, trg_tensor, trg_mask)
                logits=self.generator(out)
            next_word_logits=logits[:,-1,:]
            next_word_idx=next_word_logits.argmax(dim=-1).item()
            tgt_indices.append(next_word_idx)
            if next_word_idx==self.en_vocab.stoi['<eos>']:
                break
        trg_words=[self.en_vocab.itos[idx] for idx in tgt_indices[1:]]
        if trg_words and trg_words[-1]=='<eos>':
            trg_words.pop()
        return ' '.join(trg_words)
        