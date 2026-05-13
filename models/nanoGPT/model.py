import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = 12
        self.n_embd = n_embd
        self.dropout = dropout
        
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            print(f"WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        
    def forward(self, x):
        B, T, C = x.size() # Batch, Sequence, Dimension
        
        # In CLIP, this operation is used with a "chunk" function
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, number of heads, T, C // number of heads)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # QK^T / sqrt{d}
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf")) # Causal Mask
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # QK^T / sqrt{d} @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y = self.resid_dropout(self.c_proj(y)) # Attention 이후 projection Layer 후 Residual Layer Dropout
        return y
    
class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=False)
        self.ln_2 = LayerNorm(n_embd, bias=False)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.mlp = MLP(n_embd, dropout)
        
    def forward(self, x):
        x = self.attn(self.ln_1(x)) + x
        x = self.mlp(self.ln_2(x)) + x
        return x
    
class LayerNorm(nn.Module):
    def __init__(self, n_embd, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None
        
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab size, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),   # weight of token embedding
                wpe = nn.Embedding(config.block_size, config.n_embd),   # weight of positional embedding
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = LayerNorm(config.n_embd, bias=config.bias)
            )
        )
        # language model head
        # hidden dimension (B, T, C) -> vocab_size로 projection (B, T, vocab_size)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.transformer.wte.weight = self.lm_head.weight
        
        # init all weights
        self.apply(self._init_weights)
        for name, params in self.named_parameters():
            if name.endswith('c_proj.weight'):
                # Residual 구조에서 Block이 N개 쌓이면, 각 Block의 출력이 계속 더해지면서 분산이 누적 (레이어 수에 비례)
                # c_proj는 Residual에 더해지기 직전의 레이어라 출력을 sqrt(N)만큼 줄여주면 분산이 안정
                torch.nn.init.normal_(params, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        print("number of parameters: %.2fM" %(self.get_num_params()/1e6,))
        
        
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block_size is only {self.config.block_size}" 
    
        # idx = [24, 573, 12, 8] (vocabulary idx)
        
        # tok = self.wte(idx) [토큰 ID > 벡터]
        # [24] -> [0.2, -0.5, 0.8, ...]
        # [573] -> [0.1, 0.3, 0.4, ...]
        # [12] -> [-0.2, 0.7, 0.1, ...]
        
        # pos = self.wpe(torch.arange(T)) [위치 인덱스 -> 벡터], This is variable value based on input sentence length 
        # [0] -> [0.1, 0.3, ...]
        # [1] -> [-0.2, 0.7, ...]
        # [2] -> [0.4, 0.1, ...]
        # 1세대의 Sinusoidal Positional Embedding과 다름
        # 2세대 Learned Positional Embedding
    
        tok_emb = self.transformer.wte(idx)     # token embeddings of shape
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)     # positional embedding of shape
        
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        
        # 일반적인 언어 모델의 Label 값은 입력 토큰을 한 step만큼 뒤로민 값
        # Input: ["I", "am", "a", "boy"]
        # Label: ["am", "a", "boy", <EOS>]
        # Why? LLM은 기본적으로 Auto-Regressive 하게 동작함.
        # Cross Entropy Loss를 사용하는 이유도 Vocabulary에서 가장 확률이 높은 단어를 하나 찾는 것과 동일
        # 일반적인 Classification에서 One-Hot Vector를 Cross Entropy 하는 것과 동일하다고 보면 됨.
        if targets is not None:     # Label 값이 존재하면
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])        # note: using list [-1] to preserve the time dim
            loss = None
        
        return logits, loss
    
    