import torch
from torch import nn

from sd.attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        """
        Handles embedding for tokens within the CLIP model, including positional embeddings.

        Attributes:
            n_vocab (int): The size of the vocabulary.
            n_embd (int): The size of each embedding vector.
            n_token (int): The number of tokens (sequence length).
        """

        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        # A learnable weight matrix encodes the position information for each token
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))

    def forward(self, tokens):
        """
        Forward pass of the embedding layer combining token and positional embeddings.

        :param tokens: Tensor of token indices with shape (Batch_Size, Seq_Len).
        :return: Combined embeddings with shape (Batch_Size, Seq_Len, Dim) after adding positional information.
        """

        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        x = self.token_embedding(tokens)
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        x += self.position_embedding

        return x


class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        """
        A single layer of the CLIP model containing a self-attention mechanism followed by a feedforward network,
        typical of Transformer architectures, including normalization and residual connections.

        :param n_head: Number of attention heads.
        :param n_embd: Dimensionality of the embedding space.
        """

        super().__init__()

        # Pre-attention norm
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        # Self attention
        self.attention = SelfAttention(n_head, n_embd)
        # Pre-FNN norm
        self.layer_norm_2 = nn.LayerNorm(n_embd)
        # Feedforward layer
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        """
        Forward pass of the CLIPLayer applying self-attention and a feedforward network with residual connections.

        :param x: Input tensor of shape (Batch_Size, Seq_Len, Dim).
        :return: Output tensor of the same shape as input after processing through the layer.
        """

        # (Batch_Size, Seq_Len, Dim)
        residue = x

        # Self Attention
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layer_norm_1(x)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.attention(x, causal_mask=True)

        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        # Feedforward layer
        # Apply a feedforward layer where the hidden dimension is 4 times the embedding dimension.

        residue = x
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layer_norm_2(x)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x = self.linear_1(x)

        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x = x * torch.sigmoid(1.702 * x)  # QuickGELU activation function

        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.linear_2(x)

        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        return x


class CLIP(nn.Module):
    """
    The CLIP model encapsulates the embedding layer and multiple sequential transformer layers for processing text input,
    designed for contrastive pre-training between text and images, although this specific implementation focuses on the text part.

    """

    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layer_norm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        Processes input tokens through the embedding layer, multiple transformer layers, and a final normalization.

        :param tokens: Input tokens with shape (Batch_Size, Seq_Len).
        :return: Output tensor of the final layer with shape (Batch_Size, Seq_Len, Dim).
        """
        tokens = tokens.type(torch.long)

        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        state = self.embedding(tokens)

        # Apply encoder layers similar to the Transformer's encoder.
        for layer in self.layers:
            # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
            state = layer(state)
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.layer_norm(state)

        return output
