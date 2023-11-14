import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost
    
        """_summary_
        text_lenによる長さを考慮した損失計算

        Args:
            quantized (_type_): _description_
            inputs (_type_): _description_
            inputs_len (_type_): _description_
        """

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized, perplexity, encodings
    
    
    
class Quantizer(nn.Module):
    def __init__(self, num_embeddings,  embedding_dim):
        super(Quantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # Learnable parameters
        self.codebook = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, latent_representation):
        # Calculate Gumbel noise
    
        # Apply Gumbel softmax to obtain discrete indices
        logits = F.cosine_similarity(latent_representation.unsqueeze(1), self.codebook.unsqueeze(0), dim=-1) / self.temperature
     
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
        logits += gumbel_noise
        soft_indices = F.softmax(logits, dim=-1)
        hard_indices = torch.argmax(soft_indices, dim=-1)
     
        # Quantized representation using hard indices
        quantized_representation = self.codebook[hard_indices]

        return quantized_representation