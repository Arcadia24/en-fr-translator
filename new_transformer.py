import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SinusoidEncoding(torch.nn.Module):
    """
    Mostly copied from
    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    """

    def __init__(self, hidden_dim, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pos_embed = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        pos_embed = pos_embed.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def forward(self, x):
        """
        Adds positional embeddings to token embeddings.
        N = batch size
        L = sequence length
        E = embedding dim

        :param x: token embeddings. Shape: (N, L, E)
        :return: token_embeddings + positional embeddings. Shape: (N, L, E)
        """
        x = x + self.pos_embed[:, : x.size(1)]
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads : int, embed_size : int, context_length : int = 10, dropout : float = 0.1) -> None:
        super().__init__()
        assert embed_size % n_heads == 0, "Embedding size must be divisible by number of heads"
        self.embed_size = embed_size
        self.n_heads = n_heads
        self.head_dim = embed_size // n_heads
        
        self.fc_out = nn.Linear(embed_size, embed_size, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.fc_dropout = nn.Dropout(dropout)
        

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """MultiHeadAttention forward pass

        Args:
            x (torch.Tensor): Tensor of size (B, T, C) for batch size, sequence length, and embedding size
            mask(bool): Use the mask or not depending encoder or decoder block

        Returns:
            torch.Tensor: Tensor of size (B, T, C) for batch size, sequence length, and embedding size
        """
        B, T, C = k.shape
        # print(q.shape, k.shape, v.shape)
        q = q.view(q.shape[0], q.shape[1], self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.n_heads, C // self.n_heads).transpose(1, 2)
        # print(mask.shape, mask.dtype)
        # print(mask[0])
        # print(mask[1])
        # print(mask.shape, k.shape)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)          # B, N, T, h
        x = x.transpose(1, 2).contiguous()   # B, T, N, h
        # print(x.shape)
        x = x.view(x.shape[0], -1, C)                  # B, T, C

        # x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # x = x.transpose(1, 2).contiguous().view(B, T, C)
        x = self.fc_dropout(self.fc_out(x))
        return x
        
class FeedForward(nn.Sequential):
    def __init__(self, embed_dim: int, dropout: float) -> None:
        super().__init__(
            nn.Linear(embed_dim, embed_dim * 4, bias=False), nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim, bias=False),
            nn.Dropout(dropout),
        )
        
class Block(nn.Module):
    def __init__(self, n_head: int, embed_size: int, context_length: int, dropout: float) -> None:
        super().__init__()
        self.qkv = nn.Linear(embed_size, embed_size * 3, bias=False)
        self.mha = MultiHeadAttention(n_head, embed_size, context_length, dropout)
        self.ffwd = FeedForward(embed_size, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x = self.norm1(x)
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, C, dim=2)
        x = x + self.mha(q, k, v, mask)
        x = x + self.ffwd(self.norm2(x))
        return x
    
class Encoder(nn.Module):
    def __init__(self, vocab_size : int, n_head: int, embed_size: int, context_length: int, dropout: float, num_layers: int) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = SinusoidEncoding(embed_size, context_length)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([Block(n_head, embed_size, context_length, dropout) for _ in range(num_layers)])
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        x = self.embed(x) * math.sqrt(self.embed_size)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, n_head: int, embed_size: int, context_length: int, dropout: float) -> None:
        super().__init__()
        self.qkv = nn.Linear(embed_size, embed_size * 3, bias=False)
        self.mmha = MultiHeadAttention(n_head, embed_size, context_length, dropout)
        self.mha = MultiHeadAttention(n_head, embed_size, context_length, dropout)
        self.ffwd = FeedForward(embed_size, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.normraw = nn.LayerNorm(embed_size)
        self.normenc = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
    
    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor, trg_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x = self.norm1(x)
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, C, dim=2)
        # print(q.shape, k.shape, v.shape, trg_mask.shape)
        x = x + self.mmha(q, k, v, trg_mask)
        x = x + self.mha(self.normraw(x), self.normenc(enc_out), self.normenc(enc_out), src_mask)
        x = x + self.ffwd(self.norm3(x))
        return x
    
class Decoder(nn.Module):
    def __init__(self, vocab_size : int, n_head: int, embed_size: int, context_length: int, dropout: float, num_layers: int) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = SinusoidEncoding(embed_size, context_length)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DecoderBlock(n_head, embed_size, context_length, dropout) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor, trg_mask: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        x = self.embed(x) * math.sqrt(self.embed_size)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, trg_mask)
        return x
    
class Transformer(nn.Module):
    def __init__(self, vocab_size : int,n_head: int, embed_size: int, context_length: int, dropout: float, num_layers: int, device) -> None:
        super().__init__()
        self.encoder = Encoder(vocab_size, n_head, embed_size, context_length, dropout, num_layers).to(device)
        self.decoder = Decoder(vocab_size, n_head, embed_size, context_length, dropout, num_layers).to(device)
        self.linear = nn.Linear(embed_size, vocab_size).to(device)
        self.device = device
        self.num_parameters = sum(p.numel() for p in self.parameters())
        self.num_buffers = sum(b.numel() for b in self.buffers())
        

    def forward(self, src: torch.Tensor, trg: torch.Tensor, src_mask: torch.Tensor, trg_mask: torch.Tensor) -> torch.Tensor:
        # print("encoder")
        enc_out = self.encoder(src, src_mask)
        # print("decoder")
        x = self.decoder(trg, enc_out, src_mask, trg_mask)
        x = self.linear(x)
        return x
def causal_mask(seq_len: int) -> torch.Tensor:
    mask = torch.ones(1, seq_len, seq_len, dtype=torch.bool)
    mask = torch.tril(mask, diagonal=0)
    return mask
    
if __name__ == "__main__":
    device = torch.device("cpu")
    sos_token = torch.tensor([161], dtype=torch.int64)
    eos_token = torch.tensor([162], dtype=torch.int64)
    pad_token = torch.tensor([160], dtype=torch.int64)
    transformer = Transformer(163, 6, 390, 256, 0.2, 6, device).to(device)
    src_input = torch.tensor([161,  43,  66,   0,  68,  79,  62,  75,  65,   0,  44,  66,  62,  82,
         73,  75,  66,  80, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
        160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
        160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
        160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
        160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
        160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
        160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
        160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
        160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
        160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
        160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
        160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
        160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
        160, 160, 160, 160]).to(device).view(1, -1)
    # print(f"source = {src_input}")
    trg_input = sos_token
    src_mask = (src_input != pad_token).int() == 1
    trg_mask = causal_mask(trg_input.shape[0]).to(device)
    print(src_input.shape, trg_input.unsqueeze(0).shape, src_mask.unsqueeze(0).shape, trg_mask.unsqueeze(0).shape)
    out = transformer(src_input, trg_input.unsqueeze(0), src_mask.unsqueeze(0), trg_mask.unsqueeze(0))
    print(out.shape)