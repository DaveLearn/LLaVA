from typing import TypedDict
import torch
from huggingface_hub import PyTorchModelHubMixin


class VQAutoEncoder(torch.nn.Module):
    def __init__(self, source_dim: int = 1024, embed_dim: int = 256, num_embeddings: int = 16384):
        super().__init__()
        self.encoder = VQAdapter(source_dim=source_dim, embed_dim=embed_dim, num_embeddings=num_embeddings)
        self.decoder = torch.nn.Linear(embed_dim, source_dim, bias=True)

    def forward(self, x):
        x, quant_loss, _ = self.encoder(x)
        x = self.decoder(x)
        return x, quant_loss
    

class VQAutoEncoderConfig(TypedDict):
   source_dim: int
   embed_dim: int
   num_embeddings: int

class VQAutoEncoderModel(VQAutoEncoder, PyTorchModelHubMixin):
    def __init__(self, config: VQAutoEncoderConfig):
        super().__init__(
            source_dim=config['source_dim'],
            embed_dim=config['embed_dim'],
            num_embeddings=config['num_embeddings']
        )


class VQAdapter(torch.nn.Module):
    def __init__(self, source_dim: int = 1024, embed_dim: int = 256, num_embeddings: int = 16384):
        super().__init__()
        self.source_to_embed = torch.nn.Linear(source_dim, embed_dim)
        self.quantizer = ImgSumQuantizer(n_e=num_embeddings, e_dim=embed_dim)

    # x = [B, L, source_dim] returns ([B, L, embed_dim], quant_loss, closest_indices)
    def forward(self, x):
        x = self.source_to_embed(x)
        x_q, quant_loss, indicies = self.quantizer(x)
        return x_q, quant_loss, indicies
        



class ImgSumQuantizer(torch.nn.Module):
    """
    based on ldm VectorQuantizer2 but for single dimension
    """
    def __init__(self, n_e, e_dim, beta=0.25):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = torch.nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    # z [batch, len, e_dim]
    def forward(self, z):

        min_encoding_indices = self.get_codebook_index(z)

        z_q = self.get_codebook_entry(min_encoding_indices)
        
        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        return z_q, loss, min_encoding_indices

    # takes [N, L] -> [N, L, e_dim]
    def get_codebook_entry(self, indices):
        # get quantized latent vectors
        z_q = self.embedding(indices)

        return z_q
    
    # takes [N, L, e_dim] -> [N, L]
    def get_codebook_index(self, z):
        # reshape z -> (batch, length, channel) and flatten
        assert z.size(-1) == self.e_dim, f"expected {self.e_dim} as last dimension, got {z.size(-1)}"
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z  (avoids mat mul)
        z_flattened = z.view(-1, self.e_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        return torch.argmin(d, dim=1).reshape(z.shape[0], z.shape[1])
