
import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter
from torch.nn.utils import clip_grad_norm_


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


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention



class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(1, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')

    def forward(self, x):
        x = x.unsqueeze(1)

        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)
        out = torch.mean(out, dim=(2, 3))

        return out.squeeze(), p1, p2

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


def train_SAGAN_PFmask_simple_epoch(G, D, opt_D, opt_G, train_loader, training_method, mixing_prob, loss_f, device, change_g, change_d, epoch, train=True):
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

        #train G
        # 明日やる



        output, dec_output, _ = G(lip=lip, prev=feature, data_len=data_len, training_method=training_method, mixing_prob=mixing_prob)
        FR_output, _, _ = G(lip_tmp)

        #output_tmp2 = output.clone().detach()

        # FR_output, _, RF_enc_output = G(lip=lip_tmp)   
        
        # train G(-logit + MSE loss) fake
        B, C, T = output.shape

        d_out_real, _, _ = D(output)
        d_out_fake, _, _ = D(FR_output)

        # d_loss_real = d_out_real.mean()
        # d_loss_fake = d_out_fake.mean()

        # dec_output_loss = loss_f.mse_loss(dec_output, feature, data_len, max_len=T)
        # output_loss = loss_f.mse_loss(output, feature, data_len, max_len=T)
        
        d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
        d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
        # if change_g:
        #     g_loss = -0.001*(g_loss_fake - g_loss_real) + dec_output_loss + output_loss
        #     #g_loss = -0.001*g_loss_fake + output_loss + dec_output_loss
        # else:
        #     g_loss = output_loss + dec_output_loss


        # output_loss_all += output_loss.item()
        # dec_output_loss_all += dec_output_loss.item()
        # loss_G_all += g_loss_fake.item()

        # if train:
        #     opt_D.zero_grad()
        #     opt_G.zero_grad()
        #     g_loss.backward()
        #     clip_grad_norm_(G.parameters(), 3.0)
        #     opt_G.step()

        # del g_loss, output_loss, dec_output_loss, lip, output, enc_output, enc_output_tmp3, FR_output, g_loss_real, g_loss_fake
        # torch.cuda.empty_cache()

        #train D()

        # FR_output, _, _ = G(lip_tmp)
        # d_out_real, _, _ = D(output_tmp)
    
        # d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

        # d_out_fake, _ , _ = D(FR_output, enc_output_tmp2, data_len, T)
        # d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

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
    
        opt_G.zero_grad()
        opt_D.zero_grad()
        #print(f'loss D: {loss_D}')

        # if iter_cnt==1:
        #     breakpoint()

        loss_D.backward()
        #clip_grad_norm_(D.parameters(), 3.0)
        opt_D.step()

        loss_D_all += loss_D.item()



        # del loss_D, FR_output, d_loss_fake, d_loss_real, enc_output_tmp, enc_output_tmp2
        # torch.cuda.empty_cache()


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

    return loss_D_all, correct_cnt, wrong_cnt, change_g, change_d


