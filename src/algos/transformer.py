import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, src):
        src = src.permute(1, 0, 2)  # Change from (seq_len, batch, input_dim) to (batch, seq_len, input_dim)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)  # Change back to (seq_len, batch, input_dim)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

    def forward(self, tgt, memory, tgt_mask=None):
        tgt = tgt.permute(1, 0, 2)  # Change from (seq_len, batch, input_dim) to (batch, seq_len, input_dim)
        memory = memory.permute(1, 0, 2)  # Change from (seq_len, batch, input_dim) to (batch, seq_len, input_dim)
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        output = output.permute(1, 0, 2)  # Change back to (seq_len, batch, input_dim)
        return output

class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(input_dim, hidden_dim, num_layers, num_heads, dropout)
        self.decoder = TransformerDecoder(input_dim, hidden_dim, num_layers, num_heads, dropout)
        self.fc = nn.Linear(input_dim, output_dim)
        self.positional_encoding = PositionalEncoding(input_dim)

    def forward(self, src, tgt):
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        output = self.fc(output)
        return output


if __name__ == "__main__":
    # Example usage
    input_dim = 512
    hidden_dim = 2048
    output_dim = 10
    num_layers = 6
    num_heads = 8
    seq_length = 100
    batch_size = 32

    model = Transformer(input_dim, hidden_dim, output_dim, num_layers, num_heads)

    # Example input tensors
    src = torch.randn(seq_length, batch_size, input_dim)  # Source sequence
    tgt = torch.randn(seq_length, batch_size, input_dim)  # Target sequence

    # Forward pass
    output = model(src, tgt)
    print(output.shape)  # Output shape: (seq_length, batch_size, output_dim)
