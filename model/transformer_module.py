from torch import nn
import torch
import torch.nn.functional as F
#from d2l import torch as d2l
import math


class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]

        return embeddings

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = 2

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)
       
        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) 

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim = -1))

        #attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x

class MultiHeadAttentionRelative(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, device, dropout=0.1):
        super().__init__()
        
        self.n_heads = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.head_dim = d_model

        self.max_relative_position = 150

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc_o = nn.Linear(n_head * d_v, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_heads
        sz_b, len_q, len_k, len_v = query.size(0), query.size(1), key.size(1), value.size(1)

        residual = query
        query = self.w_qs(query).view(sz_b, len_q, n_head, d_k)
        key = self.w_ks(key).view(sz_b, len_k, n_head, d_k)
        value = self.w_vs(value).view(sz_b, len_v, n_head, d_v)
       
        r_q1 = query.view(sz_b, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(sz_b, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) 

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, sz_b*self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(sz_b, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim = -1))

        #attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(sz_b, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, sz_b*self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(sz_b, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(sz_b, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.dropout(self.fc_o(x))
        
        
        #x = [batch size, query len, hid dim]
        
        return x


class VectorQuantizer(nn.Module):
  def __init__(self, embedding_dim, conv_dim=4048, num_embeddings=512, commitment_cost=0.25):
    super(VectorQuantizer, self).__init__()

    self.pre_conv = nn.Conv1d(embedding_dim, conv_dim, kernel_size=1)
    self.post_conv = nn.Conv1d(conv_dim, embedding_dim, kernel_size=1)

    self._embedding_dim = conv_dim
    self._num_embeddings = num_embeddings
    self._commitment_cost = commitment_cost
    # コードブック(ボトルネック)
    self._w = nn.Embedding(self._num_embeddings, self._embedding_dim)
    self._w.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
     
  def forward(self, inputs):
    '''
    inputs: N×C×H×W -> (B, len, C)
    '''
    #breakpoint()
    # N×C×H×WをN×H×W×Cに変換する. (Cは埋め込みベクトルの次元)
    #inputs = inputs.permute(0, 2, 3, 1).contiguous()

    inputs = self.pre_conv(inputs.transpose(1, 2))
    #print(f'shape: {inputs.shape}')
    inputs = inputs.transpose(1, 2)

    input_shape = inputs.size()
    
    input_flattened = inputs.reshape(-1, self._embedding_dim) # すべて縦に並べる
    distances = (torch.sum(input_flattened ** 2, dim=1, keepdim=True) 
                    - 2 * torch.matmul(input_flattened, self._w.weight.t())
                    + torch.sum(self._w.weight ** 2, dim=1))
    encoding_indices = torch.argmax(-distances, 1).unsqueeze(1)
    # one-hotベクトルに変換
    encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
    encodings.scatter_(1, encoding_indices, 1) # one-hot
    # 埋め込み表現を取得し、元のインプットの形に戻す。
    quantized = torch.matmul(encodings, self._w.weight) # one-hot ⇒ ベクトル
    quantized = quantized.view(input_shape) 
     
    # 損失の計算
    # 二乗誤差で計算. sgの部分はdetach()で勾配を計算しないようにする
    e_latent_loss = F.mse_loss(quantized.detach(), inputs)
    q_latent_loss = F.mse_loss(quantized, inputs.detach())
    loss = q_latent_loss + self._commitment_cost * e_latent_loss
     
    # sgの部分はdetach()で勾配を計算しない
    quantized = inputs + (quantized - inputs).detach()
  
    #quantized = quantized.permute(0, 3, 1, 2).contiguous()
    # perplexityを計算 
    avg_probs = torch.mean(encodings, dim=0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    

    quantized = self.post_conv(quantized.transpose(1, 2))
    quantized = quantized.transpose(1, 2)
    return quantized, loss
    # return {'distances': distances,
    #         'quantize': quantized,
    #         'loss': loss, 
    #         'encodings': encodings,
    #         'encoding_indices': encoding_indices,
    #         'perplexity': perplexity}