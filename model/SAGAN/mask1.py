
import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter
from torch.nn.utils import clip_grad_norm_, weight_norm


import math

def make_pad_mask(lengths, max_len):
    """
    口唇動画,音響特徴量に対してパディングした部分を隠すためのマスク
    """
    # この後の処理でリストになるので先にdeviceを取得しておく
    device = lengths.device

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if max_len is None:
        max_len = int(max(lengths))

    seq_range = torch.arange(0, max_len, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    mask = mask.unsqueeze(1).repeat(1, max_len, 1).to(device=device)
    
    return mask


def make_mask_out(lengths, max_len):
    """
    口唇動画,音響特徴量に対してパディングした部分を隠すためのマスク
    """
    # この後の処理でリストになるので先にdeviceを取得しておく
    device = lengths.device

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if max_len is None:
        max_len = int(max(lengths))

    seq_range = torch.arange(0, max_len, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    mask = mask.unsqueeze(1).to(device=device)
    
    return mask

def resize_datalen(data_len, kernel_size, padding, stride, T):
    #print(f'before: {data_len}, {T}')
    tmp = (data_len +2*padding - (kernel_size-1)-1)/stride + 1
    T = (T +2*padding - (kernel_size-1)-1)/stride + 1
    data_len, T = torch.floor(tmp).int(), math.floor(T)
    #print(f'after: {data_len}, {T}')
    return data_len, T


def train_SAGAN_onemask_epoch(G, D, opt_D, opt_G, train_loader, training_method, mixing_prob, loss_f, device, change_g, change_d, epoch):
    iter_cnt = 0
    all_iter = len(train_loader)
    loss_G_all = 0
    loss_D_all = 0
    output_loss_all = 0
    dec_output_loss_all = 0

    correct_cnt = 0
    wrong_cnt = 0
    batch_all = 0

    print('start simple GAN re')
    print(f'change_g: {change_g}, change_d: {change_d}')

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        iter_cnt += 1

        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len, speaker = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device), speaker.to(device)

        with torch.no_grad():
            _, _, enc_output = G(lip=lip, prev=feature, data_len=data_len, training_method=training_method, mixing_prob=mixing_prob) 

        enc_output_tmp = enc_output.clone().detach()
        enc_output_tmp2 = enc_output.clone().detach()
        enc_output_tmp3 = enc_output.clone().detach()

        output, dec_output, feat_add_out = G(lip=lip, prev=feature, data_len=data_len, training_method=training_method, mixing_prob=mixing_prob)               

        lip_tmp = lip.detach()
        output_tmp = output.detach()

        # train G(-logit + MSE loss)
        B, C, T = output.shape
        g_out_fake, _, _ = D(output, enc_output, data_len, T)
        g_out_real, _, _ = D(feature, enc_output_tmp, data_len, T)

        g_loss_real = g_out_real.mean()
        g_loss_fake = g_out_fake.mean()

        #breakpoint()
        dec_output_loss = loss_f.mse_loss(dec_output, feature, data_len, max_len=T)
        output_loss = loss_f.mse_loss(output, feature, data_len, max_len=T)
        
        if change_g:
            #g_loss = -0.005*(g_loss_fake - g_loss_real) + output_loss + dec_output_loss
            g_loss = -0.0001*g_loss_fake + output_loss + dec_output_loss
        else:
            g_loss = dec_output_loss + output_loss

        output_loss_all += output_loss.item()
        dec_output_loss_all += dec_output_loss.item()
        loss_G_all += g_loss_fake.item()

        opt_D.zero_grad()
        opt_G.zero_grad()
        g_loss.backward()

        clip_grad_norm_(G.parameters(), 3.0)
        opt_G.step()
        
        del g_loss, output_loss, dec_output_loss, enc_output, enc_output_tmp, g_loss_real, g_loss_fake
        torch.cuda.empty_cache()


        #train D()
        output, _, _ = G(lip=lip_tmp, prev=feature, data_len=data_len, training_method=training_method, mixing_prob=mixing_prob)

        d_out_real, _, _ = D(feature, enc_output_tmp2, data_len, T)
        d_out_fake, _, _ = D(output, enc_output_tmp3, data_len, T)
      
        d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
        d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

        loss_D = d_loss_fake + d_loss_real

        for i in range(d_out_real.shape[0]):
            prob = d_out_real[i]

            if prob>=0:
                correct_cnt += 1
            else:
                wrong_cnt += 1

            batch_all += 1

        for i in range(d_out_fake.shape[0]):
            prob = d_out_fake[i]

            if prob<0:
                correct_cnt += 1
            else:
                wrong_cnt += 1

            batch_all += 1

        if change_d:
            opt_G.zero_grad()
            opt_D.zero_grad()
            loss_D.backward()
            clip_grad_norm_(D.parameters(), 3.0)
            opt_D.step()

        loss_D_all += loss_D.item()

        del loss_D, d_loss_fake, d_loss_real, output_tmp, feature
        torch.cuda.empty_cache()

        if True:
            if iter_cnt > 200:
                break
        # if iter_cnt > 10:
        #     break
    output_loss_all /= all_iter
    dec_output_loss_all /= all_iter
    loss_G_all /= all_iter
    loss_D_all /= all_iter

    correct_cnt /= batch_all
    wrong_cnt /= batch_all

    if correct_cnt>=0.6:
        change_g = True
    else:
        change_g = False

    if correct_cnt<0.9:
        change_d = True
    else:
        change_d = False

    return output_loss_all, dec_output_loss_all, loss_G_all, loss_D_all, correct_cnt, wrong_cnt, change_g, change_d

def train_SAGAN_PFmask_epoch(G, D, opt_D, opt_G, train_loader, training_method, mixing_prob, loss_f, device, change_g, change_d, epoch, train=True):
    iter_cnt = 0
    all_iter = len(train_loader)
    loss_G_all = 0
    loss_D_all = 0
    output_loss_all = 0
    dec_output_loss_all = 0

    correct_cnt = 0
    wrong_cnt = 0
    batch_all = 0

    print('start Professor learning')
    print(f'change_g: {change_g}, change_d: {change_d}')


    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        iter_cnt += 1
        
        lip, feature, feat_add, upsample, data_len, speaker, label = batch

        lip, feature, data_len = lip.to(device), feature.to(device), data_len.to(device)
        lip_tmp = lip.clone().detach()
        lip_tmp2 = lip.clone().detach()

        with torch.no_grad():
            _, _, enc_output = G(lip=lip, prev=feature, data_len=data_len, training_method=training_method, mixing_prob=mixing_prob) 
        
        enc_output_tmp = enc_output.clone().detach()
        enc_output_tmp2 = enc_output.clone().detach()
        enc_output_tmp3 = enc_output.clone().detach()

        output, dec_output, _ = G(lip=lip, prev=feature, data_len=data_len, training_method=training_method, mixing_prob=mixing_prob)
        FR_output, _, _ = G(lip_tmp)

        output_tmp = output.clone().detach()
        #output_tmp2 = output.clone().detach()

        # FR_output, _, RF_enc_output = G(lip=lip_tmp)   
        
        # train G(-logit + MSE loss) fake
        B, C, T = output.shape

        g_out_real, _, _ = D(output, enc_output, data_len, T)
        g_out_fake, _, _ = D(FR_output, enc_output_tmp3, data_len, T)

        g_loss_real = g_out_real.mean()
        g_loss_fake = g_out_fake.mean()

        dec_output_loss = loss_f.mse_loss(dec_output, feature, data_len, max_len=T)
        output_loss = loss_f.mse_loss(output, feature, data_len, max_len=T)
        
        if change_g:
            g_loss = -0.001*(g_loss_fake - g_loss_real) + dec_output_loss + output_loss
            #g_loss = -0.001*g_loss_fake + output_loss + dec_output_loss
        else:
            g_loss = output_loss + dec_output_loss


        output_loss_all += output_loss.item()
        dec_output_loss_all += dec_output_loss.item()
        loss_G_all += g_loss_fake.item()

        if train:
            opt_D.zero_grad()
            opt_G.zero_grad()
            g_loss.backward()
            clip_grad_norm_(G.parameters(), 3.0)
            opt_G.step()

        del g_loss, output_loss, dec_output_loss, lip, output, enc_output, enc_output_tmp3, FR_output, g_loss_real, g_loss_fake
        torch.cuda.empty_cache()

        #train D()

        FR_output, _, enc_output = G(lip_tmp)
        d_out_real, _, _ = D(output_tmp, enc_output_tmp, data_len, T)
    
        d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

        d_out_fake, _ , _ = D(FR_output, enc_output_tmp2, data_len, T)
        d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

        for i in range(d_out_real.shape[0]):
            prob = d_out_real[i]
            print(f'real: {prob}')
            if prob>=0:
                correct_cnt += 1
            else:
                wrong_cnt += 1

            batch_all += 1

        for i in range(d_out_fake.shape[0]):
            prob = d_out_fake[i]
            print(f'fake: {prob}')
            if prob<0:
                correct_cnt += 1
            else:
                wrong_cnt += 1

            batch_all += 1

        loss_D = d_loss_fake + d_loss_real 
        if change_d:
            opt_G.zero_grad()
            opt_D.zero_grad()
            #print(f'loss D: {loss_D}')

            # if iter_cnt==1:
            #     breakpoint()

            loss_D.backward()
            #clip_grad_norm_(D.parameters(), 3.0)
            opt_D.step()

        loss_D_all += loss_D.item()



        del loss_D, FR_output, d_loss_fake, d_loss_real, enc_output_tmp, enc_output_tmp2
        torch.cuda.empty_cache()


        # if iter_cnt > 10:
        #     break

        if True:
            if iter_cnt > 200:
                break

    output_loss_all /= all_iter
    dec_output_loss_all /= all_iter
    loss_G_all /= all_iter
    loss_D_all /= all_iter

    correct_cnt /= batch_all
    wrong_cnt /= batch_all

    if correct_cnt>=0.6:
        change_g = True
    else:
        change_g = False

    if correct_cnt<0.9:
        change_d = True
    else:
        change_d = False

    return output_loss_all, dec_output_loss_all, loss_G_all, loss_D_all, correct_cnt, wrong_cnt, change_g, change_d


def train_SAGAN_PFmask_simple_epoch(G, D, opt_D, opt_G, train_loader, training_method, mixing_prob, loss_f, device, change_g, change_d, epoch, train=False):
    iter_cnt = 0
    all_iter = len(train_loader)
    loss_G_all = 0
    loss_D_all = 0
    output_loss_all = 0
    dec_output_loss_all = 0

    correct_cnt = 0
    wrong_cnt = 0
    batch_all = 0

    print('start Professor learning')
    print(f'change_g: {change_g}, change_d: {change_d}')


    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        iter_cnt += 1
        

        #train D
        lip, feature, feat_add, upsample, data_len, speaker, label = batch

        lip, feature, data_len = lip.to(device), feature.to(device), data_len.to(device)
        lip_tmp = lip.clone().detach()
        output, _, _ = G(lip=lip, prev=feature, data_len=data_len, training_method=training_method, mixing_prob=mixing_prob)
        FR_output, _, _ = G(lip_tmp)
        B, C, T = output.shape

        d_out_real, _, _ = D(output, data_len, T)
        d_out_fake, _, _ = D(FR_output, data_len, T)

        d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
        d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

        #output_tmp2 = output.clone().detach()

        # FR_output, _, RF_enc_output = G(lip=lip_tmp)   
        
        # train G(-logit + MSE loss) fake
        B, C, T = output.shape

        d_out_real, _, _ = D(output, data_len, T)
        d_out_fake, _, _ = D(FR_output, data_len, T)

        # d_loss_real = d_out_real.mean()
        # d_loss_fake = d_out_fake.mean()

        # dec_output_loss = loss_f.mse_loss(dec_output, feature, data_len, max_len=T)
        # output_loss = loss_f.mse_loss(output, feature, data_len, max_len=T)
        
        d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
        d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

        loss_D = d_loss_fake + d_loss_real
        if change_d:
            opt_G.zero_grad()
            opt_D.zero_grad()
            loss_D.backward()
            clip_grad_norm_(D.parameters(), 3.0)
            opt_D.step()

        loss_D_all += loss_D.item()
        
        for i in range(d_out_real.shape[0]):
            prob = d_out_real[i]
            print(f'real: {prob}')
            if prob>=0:
                correct_cnt += 1
            else:
                wrong_cnt += 1

            batch_all += 1

        for i in range(d_out_fake.shape[0]):
            prob = d_out_fake[i]
            print(f'fake: {prob}')
            if prob<0:
                correct_cnt += 1
            else:
                wrong_cnt += 1

            batch_all += 1


        
        del loss_D, FR_output, d_loss_fake, d_loss_real
        torch.cuda.empty_cache()



        #train G
        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, data_len = lip.to(device), feature.to(device), data_len.to(device)

        lip_tmp = lip.clone().detach()
        output, dec_output, _ = G(lip=lip, prev=feature, data_len=data_len, training_method=training_method, mixing_prob=mixing_prob)
        FR_output, _, _ = G(lip_tmp)

        g_out_real, _, _ = D(output, data_len, T)
        g_out_fake, _, _ = D(FR_output, data_len, T)

        g_loss_real = g_out_real.mean()
        g_loss_fake = g_out_fake.mean()

        dec_output_loss = loss_f.mse_loss(dec_output, feature, data_len, max_len=T)
        output_loss = loss_f.mse_loss(output, feature, data_len, max_len=T)
        

        if change_g:
            g_loss = -0.001*(g_loss_fake - g_loss_real) + dec_output_loss + output_loss
            #g_loss = -0.001*g_loss_fake + output_loss + dec_output_loss
        else:
            g_loss = output_loss + dec_output_loss


        output_loss_all += output_loss.item()
        dec_output_loss_all += dec_output_loss.item()
        loss_G_all += g_loss_fake.item()

        # if train:
        opt_D.zero_grad()
        opt_G.zero_grad()
        g_loss.backward()
        clip_grad_norm_(G.parameters(), 3.0)
        opt_G.step()

        # del g_loss, output_loss, dec_output_loss, lip, output, enc_output, enc_output_tmp3, FR_output, g_loss_real, g_loss_fake
        # torch.cuda.empty_cache()

        #train D()

        # FR_output, _, _ = G(lip_tmp)
        # d_out_real, _, _ = D(output_tmp)
    
        # d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

        # d_out_fake, _ , _ = D(FR_output, enc_output_tmp2, data_len, T)
        # d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

        # for i in range(d_out_real.shape[0]):
        #     prob = d_out_real[i]
        #     print(f'real: {prob}')
        #     if prob>=0:
        #         correct_cnt += 1
        #     else:
        #         wrong_cnt += 1

        #     batch_all += 1

        # for i in range(d_out_fake.shape[0]):
        #     prob = d_out_fake[i]
        #     print(f'fake: {prob}')
        #     if prob<0:
        #         correct_cnt += 1
        #     else:
        #         wrong_cnt += 1

        #     batch_all += 1

        # loss_D = d_loss_fake + d_loss_real 
        # if change_d:
        #     opt_G.zero_grad()
        #     opt_D.zero_grad()
        #     #print(f'loss D: {loss_D}')

        #     # if iter_cnt==1:
        #     #     breakpoint()

        #     loss_D.backward()
        #     #clip_grad_norm_(D.parameters(), 3.0)
        #     opt_D.step()

        # loss_D_all += loss_D.item()



        # # del loss_D, FR_output, d_loss_fake, d_loss_real, enc_output_tmp, enc_output_tmp2
        # # torch.cuda.empty_cache()


        # if iter_cnt > 10:
        #     break

        if True:
            if iter_cnt > 200:
                break

    output_loss_all /= all_iter
    dec_output_loss_all /= all_iter
    loss_G_all /= all_iter
    loss_D_all /= all_iter

    correct_cnt /= batch_all
    wrong_cnt /= batch_all

    if correct_cnt>=0.6:
        change_g = True
    else:
        change_g = False

    if correct_cnt<0.9:
        change_d = True
    else:
        change_d = False

    return output_loss_all, dec_output_loss_all, loss_G_all, loss_D_all, correct_cnt, wrong_cnt, change_g, change_d



class Self_Attn_suvey_mask(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super().__init__()

        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x, data_mask):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        #print(f'att x:{x.shape}, mask:{data_mask.shape}')
    
        m_batchsize,C, L = x.size()
        proj_query  = self.query_conv(x).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x) # B X C x (*W*H)
     

        energy =  torch.bmm(proj_query,proj_key) # transpose check
        mask = torch.triu(torch.full((L, L), True), diagonal=1).to(x.device).unsqueeze(0).repeat(m_batchsize, 1, 1)

        mask = mask | data_mask
        energy1 = energy.masked_fill(mask, torch.tensor(float('-inf')))
        #breakpoint()
        #energy1 = torch.dot(energy, tgt_mask)
        
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        # out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)



class Discriminator_suvey_test3(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=4, image_size=64, conv_dim=512, kernel_size=4, padding=1,stride=2, PL=False):
        super().__init__()
        self.imsize = image_size
        self.PL = PL
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv1d(80+256, 512, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if PL:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv1d(curr_dim, 1,  kernel_size=self.kernel_size, padding=self.padding, stride=self.stride))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn_suvey_mask(2048, 'relu')
        if self.PL:
            self.attn2 = Self_Attn_suvey_mask(4096, 'relu')
        else:
            self.attn2 = Self_Attn_suvey_mask(2048, 'relu')

        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.project = nn.Conv1d(256, 80, kernel_size=1)

        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.drop3 = nn.Dropout(p=0.5)

        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(80)

        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x, enc_output, data_len, max_len):
        
        # noise2 = (torch.rand(x.shape) - 0.5)
        # noise2 = noise2.to(x.device)
        # #breakpoint()
        # data_mask = make_pad_mask(data_len, max_len)
        # out_mask = make_mask_out(data_len, max_len)

        enc_output = enc_output.permute(0, 2, 1)
        enc_output = self.upsampling(enc_output)

        #noise2 = (torch.rand(enc_output.shape) - 0.5) / 0.5

        # enc_output = self.project(enc_output)
        # enc_output = self.relu(enc_output)
        # noise1 = (torch.rand(enc_output.shape) - 0.5)
        # noise1 = noise1.to(x.device)
 
        enc_output = self.batchnorm1(enc_output)
        x = self.batchnorm2(x)
       
        # enc_output += noise1
        # x += noise2
       
        x = torch.cat((x, enc_output), dim=1)
        print(f'cat min:{x.min()}, max:{x.max()}')
        # enc_output = enc_output.unsqueeze(1)

        # x = x.unsqueeze(1)


        # x = torch.cat((x, enc_output), dim=1)
        # max_len = torch.tensor(10)
        # data = [3, 4, 5, 10]

        # data = torch.tensor(data)
        # mask = make_pad_mask(data_len, max_len)
        # mask_out = make_mask_out(data_len, max_len)
        
        out = self.l1(x)
        print(f'l1 min:{out.min()}, max:{out.max()}')
        data_len, max_len = resize_datalen(data_len, self.kernel_size, self.padding, self.stride, max_len)
        #print(f'l1: {out.shape}, {max_len}')
    
        #out = self.drop3(out)

        out = self.l2(out)
        print(f'l2 min:{out.min()}, max:{out.max()}')
        data_len, max_len = resize_datalen(data_len, self.kernel_size, self.padding, self.stride, max_len)
        #print(f'l2: {out.shape}, {max_len}')

        out = self.l3(out)
        print(f'l3 min:{out.min()}, max:{out.max()}')
        data_len, max_len = resize_datalen(data_len, self.kernel_size, self.padding, self.stride, max_len)
        #print(f'l3: {out.shape}, {max_len}')

        mask = make_pad_mask(data_len, max_len)
        out,p1 = self.attn1(out, mask)
        print(f'att1 min:{out.min()}, max:{out.max()}')
        if self.PL:
            out=self.l4(out)
            data_len, max_len = resize_datalen(data_len, self.kernel_size, self.padding, self.stride, max_len)

        mask = make_pad_mask(data_len, max_len)
        out,p2 = self.attn2(out, mask)
        print(f'att2 min:{out.min()}, max:{out.max()}')

        out=self.last(out)
        print(f'last min:{out.min()}, max:{out.max()}')
        data_len, max_len = resize_datalen(data_len, self.kernel_size, self.padding, self.stride, max_len)
        


        
        #(写真ごとに確率算出)
        mask_out = make_mask_out(data_len, max_len)
        out1 = out.masked_fill(mask_out, torch.tensor(0))
        out1 = torch.mean(out1, dim=(2))
        print(f'out1 min:{out1.min()}, max:{out.max()}')

        return out1.squeeze(), p1, p2


class Discriminator_suvey_simple(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=4, image_size=64, conv_dim=512, kernel_size=4, padding=1,stride=2, PL=False):
        super().__init__()
        self.imsize = image_size
        self.PL = PL
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv1d(80, 160, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = 160

        layer2.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        # layer3.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)))
        # layer3.append(nn.LeakyReLU(0.1))
        # curr_dim = curr_dim * 2

        if PL:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)


        self.attn1 = Self_Attn_suvey_mask(320, 'relu')
        if self.PL:
            self.attn2 = Self_Attn_suvey_mask(640, 'relu')
            last.append(weight_norm(nn.Conv1d(640, 1,  kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)))
        else:
            self.attn2 = Self_Attn_suvey_mask(320, 'relu')
            last.append(weight_norm(nn.Conv1d(320, 1,  kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)))
            
        self.last = nn.Sequential(*last)

        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.project = nn.Conv1d(256, 80, kernel_size=1)

        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.drop3 = nn.Dropout(p=0.5)

        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(80)

        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x, data_len, max_len):
        
        # noise2 = (torch.rand(x.shape) - 0.5)
        # noise2 = noise2.to(x.device)
        # #breakpoint()
        # data_mask = make_pad_mask(data_len, max_len)
        # out_mask = make_mask_out(data_len, max_len)

        # enc_output = enc_output.permute(0, 2, 1)
        # enc_output = self.upsampling(enc_output)

        #noise2 = (torch.rand(enc_output.shape) - 0.5) / 0.5

        # enc_output = self.project(enc_output)
        # enc_output = self.relu(enc_output)
        # noise1 = (torch.rand(enc_output.shape) - 0.5)
        # noise1 = noise1.to(x.device)
 
        # enc_output = self.batchnorm1(enc_output)
        # x = self.batchnorm2(x)
       
        # # enc_output += noise1
        # # x += noise2
       
        # x = torch.cat((x, enc_output), dim=1)
        print(f'cat min:{x.min()}, max:{x.max()}')
        # enc_output = enc_output.unsqueeze(1)

        # x = x.unsqueeze(1)


        # x = torch.cat((x, enc_output), dim=1)
        # max_len = torch.tensor(10)
        # data = [3, 4, 5, 10]

        # data = torch.tensor(data)
        # mask = make_pad_mask(data_len, max_len)
        # mask_out = make_mask_out(data_len, max_len)
        
        out = self.l1(x)
        print(f'l1 min:{out.min()}, max:{out.max()}')
        data_len, max_len = resize_datalen(data_len, self.kernel_size, self.padding, self.stride, max_len)
        #print(f'l1: {out.shape}, {max_len}')
    
        #out = self.drop3(out)

        out = self.l2(out)
        print(f'l2 min:{out.min()}, max:{out.max()}')
        data_len, max_len = resize_datalen(data_len, self.kernel_size, self.padding, self.stride, max_len)
        #print(f'l2: {out.shape}, {max_len}')

        # out = self.l3(out)
        # print(f'l3 min:{out.min()}, max:{out.max()}')
        # data_len, max_len = resize_datalen(data_len, self.kernel_size, self.padding, self.stride, max_len)
        #print(f'l3: {out.shape}, {max_len}')

        mask = make_pad_mask(data_len, max_len)
        out,p1 = self.attn1(out, mask)
        print(f'att1 min:{out.min()}, max:{out.max()}')
        if self.PL:
            out=self.l4(out)
            data_len, max_len = resize_datalen(data_len, self.kernel_size, self.padding, self.stride, max_len)

        #mask = make_pad_mask(data_len, max_len)
        #out,p2 = self.attn2(out, mask)
        print(f'att2 min:{out.min()}, max:{out.max()}')

        out=self.last(out)
        print(f'last min:{out.min()}, max:{out.max()}')
        data_len, max_len = resize_datalen(data_len, self.kernel_size, self.padding, self.stride, max_len)
        


        
        #(写真ごとに確率算出)
        mask_out = make_mask_out(data_len, max_len)
        out1 = out.masked_fill(mask_out, torch.tensor(0))
        out1 = torch.mean(out1, dim=(2))
        print(f'out1 min:{out1.min()}, max:{out.max()}')

        return out1.squeeze(), p1, p1


class Discriminator_suvey_simple(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=4, image_size=64, conv_dim=512, kernel_size=4, padding=1,stride=2, PL=False):
        super().__init__()
        self.imsize = image_size
        self.PL = PL
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv1d(80, 160, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = 160

        layer2.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if PL:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)


        self.attn1 = Self_Attn_suvey_mask(320, 'relu')
        if self.PL:
            self.attn2 = Self_Attn_suvey_mask(640, 'relu')
            last.append(nn.Conv1d(640, 1,  kernel_size=self.kernel_size, padding=self.padding, stride=self.stride))
        else:
            self.attn2 = Self_Attn_suvey_mask(320, 'relu')
            last.append(nn.Conv1d(320, 1,  kernel_size=self.kernel_size, padding=self.padding, stride=self.stride))
            
        self.last = nn.Sequential(*last)

        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.project = nn.Conv1d(256, 80, kernel_size=1)

        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.drop3 = nn.Dropout(p=0.5)

        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(80)

        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x, data_len, max_len):
        
        # noise2 = (torch.rand(x.shape) - 0.5)
        # noise2 = noise2.to(x.device)
        # #breakpoint()
        # data_mask = make_pad_mask(data_len, max_len)
        # out_mask = make_mask_out(data_len, max_len)

        # enc_output = enc_output.permute(0, 2, 1)
        # enc_output = self.upsampling(enc_output)

        #noise2 = (torch.rand(enc_output.shape) - 0.5) / 0.5

        # enc_output = self.project(enc_output)
        # enc_output = self.relu(enc_output)
        # noise1 = (torch.rand(enc_output.shape) - 0.5)
        # noise1 = noise1.to(x.device)
 
        # enc_output = self.batchnorm1(enc_output)
        # x = self.batchnorm2(x)
       
        # # enc_output += noise1
        # # x += noise2
       
        # x = torch.cat((x, enc_output), dim=1)
        print(f'cat min:{x.min()}, max:{x.max()}')
        # enc_output = enc_output.unsqueeze(1)

        # x = x.unsqueeze(1)


        # x = torch.cat((x, enc_output), dim=1)
        # max_len = torch.tensor(10)
        # data = [3, 4, 5, 10]

        # data = torch.tensor(data)
        # mask = make_pad_mask(data_len, max_len)
        # mask_out = make_mask_out(data_len, max_len)
        
        out = self.l1(x)
        print(f'l1 min:{out.min()}, max:{out.max()}')
        data_len, max_len = resize_datalen(data_len, self.kernel_size, self.padding, self.stride, max_len)
        #print(f'l1: {out.shape}, {max_len}')
    
        #out = self.drop3(out)

        out = self.l2(out)
        print(f'l2 min:{out.min()}, max:{out.max()}')
        data_len, max_len = resize_datalen(data_len, self.kernel_size, self.padding, self.stride, max_len)
        #print(f'l2: {out.shape}, {max_len}')

        # out = self.l3(out)
        # print(f'l3 min:{out.min()}, max:{out.max()}')
        # data_len, max_len = resize_datalen(data_len, self.kernel_size, self.padding, self.stride, max_len)
        #print(f'l3: {out.shape}, {max_len}')

        mask = make_pad_mask(data_len, max_len)
        out,p1 = self.attn1(out, mask)
        print(f'att1 min:{out.min()}, max:{out.max()}')
        if self.PL:
            out=self.l4(out)
            data_len, max_len = resize_datalen(data_len, self.kernel_size, self.padding, self.stride, max_len)

        #mask = make_pad_mask(data_len, max_len)
        #out,p2 = self.attn2(out, mask)
        print(f'att2 min:{out.min()}, max:{out.max()}')

        out=self.last(out)
        print(f'last min:{out.min()}, max:{out.max()}')
        data_len, max_len = resize_datalen(data_len, self.kernel_size, self.padding, self.stride, max_len)
        


        
        #(写真ごとに確率算出)
        mask_out = make_mask_out(data_len, max_len)
        out1 = out.masked_fill(mask_out, torch.tensor(0))
        out1 = torch.mean(out1, dim=(2))
        print(f'out1 min:{out1.min()}, max:{out.max()}')

        return out1.squeeze(), p1, p1
        

class Discriminator_suvey_simple_2d(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=4, image_size=64, conv_dim=512, kernel_size=4, padding=1,stride=2, PL=False):
        super().__init__()
        self.imsize = image_size
        self.PL = PL
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv1d(80, 160, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = 160

        layer2.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if PL:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv1d(curr_dim, 1,  kernel_size=self.kernel_size, padding=self.padding, stride=self.stride))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn_suvey_mask(640, 'relu')
        if self.PL:
            self.attn2 = Self_Attn_suvey_mask(1280, 'relu')
        else:
            self.attn2 = Self_Attn_suvey_mask(640, 'relu')

        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.project = nn.Conv1d(256, 80, kernel_size=1)

        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.drop3 = nn.Dropout(p=0.5)

        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(80)

        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x, data_len, max_len):
        
        # noise2 = (torch.rand(x.shape) - 0.5)
        # noise2 = noise2.to(x.device)
        # #breakpoint()
        # data_mask = make_pad_mask(data_len, max_len)
        # out_mask = make_mask_out(data_len, max_len)

        # enc_output = enc_output.permute(0, 2, 1)
        # enc_output = self.upsampling(enc_output)

        #noise2 = (torch.rand(enc_output.shape) - 0.5) / 0.5

        # enc_output = self.project(enc_output)
        # enc_output = self.relu(enc_output)
        # noise1 = (torch.rand(enc_output.shape) - 0.5)
        # noise1 = noise1.to(x.device)
 
        # enc_output = self.batchnorm1(enc_output)
        # x = self.batchnorm2(x)
       
        # # enc_output += noise1
        # # x += noise2
       
        # x = torch.cat((x, enc_output), dim=1)
        print(f'cat min:{x.min()}, max:{x.max()}')
        # enc_output = enc_output.unsqueeze(1)

        # x = x.unsqueeze(1)


        # x = torch.cat((x, enc_output), dim=1)
        # max_len = torch.tensor(10)
        # data = [3, 4, 5, 10]

        # data = torch.tensor(data)
        # mask = make_pad_mask(data_len, max_len)
        # mask_out = make_mask_out(data_len, max_len)
        
        out = self.l1(x)
        print(f'l1 min:{out.min()}, max:{out.max()}')
        data_len, max_len = resize_datalen(data_len, self.kernel_size, self.padding, self.stride, max_len)
        #print(f'l1: {out.shape}, {max_len}')
    
        #out = self.drop3(out)

        out = self.l2(out)
        print(f'l2 min:{out.min()}, max:{out.max()}')
        data_len, max_len = resize_datalen(data_len, self.kernel_size, self.padding, self.stride, max_len)
        #print(f'l2: {out.shape}, {max_len}')

        out = self.l3(out)
        print(f'l3 min:{out.min()}, max:{out.max()}')
        data_len, max_len = resize_datalen(data_len, self.kernel_size, self.padding, self.stride, max_len)
        #print(f'l3: {out.shape}, {max_len}')

        mask = make_pad_mask(data_len, max_len)
        out,p1 = self.attn1(out, mask)
        print(f'att1 min:{out.min()}, max:{out.max()}')
        if self.PL:
            out=self.l4(out)
            data_len, max_len = resize_datalen(data_len, self.kernel_size, self.padding, self.stride, max_len)

        mask = make_pad_mask(data_len, max_len)
        out,p2 = self.attn2(out, mask)
        print(f'att2 min:{out.min()}, max:{out.max()}')

        out=self.last(out)
        print(f'last min:{out.min()}, max:{out.max()}')
        data_len, max_len = resize_datalen(data_len, self.kernel_size, self.padding, self.stride, max_len)
        


        
        #(写真ごとに確率算出)
        mask_out = make_mask_out(data_len, max_len)
        out1 = out.masked_fill(mask_out, torch.tensor(0))
        out1 = torch.mean(out1, dim=(2))
        print(f'out1 min:{out1.min()}, max:{out.max()}')

        return out1.squeeze(), p1, p2


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)