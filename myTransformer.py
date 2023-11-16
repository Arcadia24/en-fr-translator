import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size : int, n_heads : int, context_length : int = 10, dropout : float = 0.2, put_mask : bool = True) -> None:
        super().__init__()
        assert embed_size % n_heads == 0, "Embedding size must be divisible by number of heads"
        self.embed_size = embed_size
        self.n_heads = n_heads
        self.head_dim = embed_size // n_heads
        self.put_mask = put_mask
        
        self.fc_out = nn.Linear(embed_size, embed_size)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.fc_dropout = nn.Dropout(dropout)
        
        self.register_buffer('mask', torch.tril(torch.ones(context_length, context_length).view(1, 1, context_length, context_length))) if put_mask else None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """MultiHeadAttention forward pass

        Args:
            x (torch.Tensor): Tensor of size (B, T, C) for batch size, sequence length, and embedding size
            mask(bool): Use the mask or not depending encoder or decoder block

        Returns:
            torch.Tensor: Tensor of size (B, T, C) for batch size, sequence length, and embedding size
        """
        B, T, C = q.shape
        
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5) 
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf")) if self.put_mask else attn
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        x = (attn @ v)                       # B, N, T, h
        x = x.transpose(1, 2).contiguous()   # B, T, N, h
        x = x.view(B, T, C)                  # B, T, C
        x = self.fc_dropout(self.fc_out(x))
        return x
        
class FeedForward(nn.Sequential):
    def __init__(self, embed_dim: int, dropout: float) -> None:
        super().__init__(
            nn.Linear(embed_dim, embed_dim * 4, bias=False), nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim, bias=False),
            nn.Dropout(dropout),
        )
        
class EncoderBlock(nn.Module):
    def __init__(self, n_head: int, embed_size: int, context_length: int, dropout: float) -> None:
        super().__init__()
        self.qkv = nn.Linear(embed_size, embed_size * 3, bias=False)
        self.mha = MultiHeadAttention(n_head, embed_size, context_length, dropout)
        self.ffwd = FeedForward(embed_size, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x = self.norm1(x)
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, C, dim=2)
        x = x + self.mha(q, k, v)
        x = x + self.ffwd(self.norm2(x))
        return x
    
class Encoder(nn.Module):
    def __init__(self, vocab_size : int, n_head: int, embed_size: int, context_length: int, dropout: float, num_layers: int) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(context_length, embed_size)
        self.layers = nn.Sequential(*[EncoderBlock(n_head, embed_size, context_length, dropout) for _ in range(num_layers)])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        x = self.dropout(self.word_embedding(x) + self.positional_emb.weight[:T])
        x = self.layers(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, n_head: int, embed_size: int, context_length: int, dropout: float) -> None:
        self.norm1 = nn.LayerNorm(embed_size)
        self.qkv1 = nn.Linear(embed_size, embed_size * 3, bias=False)
        self.mham = MultiHeadAttention(n_head, embed_size, context_length, dropout, False)
        self.mha = MultiHeadAttention(n_head, embed_size, context_length, dropout)
        self.ffwd = FeedForward(embed_size, dropout)
        self.normenc = nn.LayerNorm(embed_size)
        self.normraw = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
    
    def forward(self, x: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x = self.norm1(x)
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, C, dim=2)
        x = x + self.mham(q, k, v)
        x = x + self.mha(self.normraw(x), self.normenc(enc_out), self.normenc(enc_out))
        x = x + self.ffwd(self.norm3(x))
        return x
    
class Decoder(nn.Module):
    def __init__(self, vocab_size : int, n_head: int, embed_size: int, context_length: int, dropout: float, num_layers: int) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(context_length, embed_size)
        self.layers = nn.Sequential(*[DecoderBlock(n_head, embed_size, context_length, dropout) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        x = self.dropout(self.word_embedding(x) + self.positional_emb.weight[:T])
        x = self.layers(x)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size : int, n_head: int, embed_size: int, context_length: int, dropout: float, num_layers: int) -> None:
        super().__init__()
        self.encoder = Encoder(vocab_size, n_head, embed_size, context_length, dropout, num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.fc_out(x)
        return x

class Block(nn.Module):
    def __init__(self, n_head: int, embed_size: int, context_length: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.qkv = nn.Linear(embed_size, embed_size * 3, bias=False)
        self.mham = MultiHeadAttention(embed_size, n_head, context_length, dropout, True)
        self.ffwd = FeedForward(embed_size, dropout)
        self.norm2 = nn.LayerNorm(embed_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x = self.norm1(x)
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, C, dim=2)
        x = x + self.mham(q, k, v)
        x = x + self.ffwd(self.norm2(x))
        return x
    
class Transformer(nn.Module):
    def __init__(self, vocab_size : int, n_head: int, embed_size: int, context_length: int, dropout: float, num_layers: int) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(context_length, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.Sequential(*[Block(n_head, embed_size, context_length, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        x = self.dropout(self.word_embedding(x) + self.pos_embedding.weight[:T])
        x = self.layers(x)
        x = self.lm_head(self.norm(x))
        return x