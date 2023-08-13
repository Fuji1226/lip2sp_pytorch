
import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter
from torch.nn.utils import clip_grad_norm_

from .mask import make_pad_mask, make_mask_out

def create_mask(B, width, height):
    len = width * height
    mask = torch.ones((len, len))

    for i in range(len):
        tmp = ((i//width)+1)*width
        #breakpoint()
        for j in range(tmp):
            mask[i, j] *= 0

    mask = mask.repeat(B, 1, 1)
    return mask


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

        mask = create_mask(m_batchsize, width, height).bool().to(x.device)
        energy = energy.masked_fill(mask, torch.tensor(float('-inf')))

        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention


class Self_Attn_suvey(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn_suvey,self).__init__()

        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
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
    
        m_batchsize,C, L = x.size()
        proj_query  = self.query_conv(x).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x) # B X C x (*W*H)
     

        energy =  torch.bmm(proj_query,proj_key) # transpose check
        mask = torch.triu(torch.full((L, L), True), diagonal=1).to(x.device).unsqueeze(0).repeat(m_batchsize, 1, 1)
        energy1 = energy.masked_fill(mask, torch.tensor(float('-inf')))
        #breakpoint()
        #energy1 = torch.dot(energy, tgt_mask)
        
        attention = self.softmax(energy1) # BX (N) X (N) 
        proj_value = self.value_conv(x) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        # out = out.view(m_batchsize,C,width,height)
        
        # out = self.gamma*out + x
        return out,attention


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
    
        m_batchsize,C, L = x.size()
        proj_query  = self.query_conv(x).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x) # B X C x (*W*H)
     

        energy =  torch.bmm(proj_query,proj_key) # transpose check
        mask = torch.triu(torch.full((L, L), True), diagonal=1).to(x.device).unsqueeze(0).repeat(m_batchsize, 1, 1)

        mask = mask | data_mask
        energy1 = energy.masked_fill(mask, torch.tensor(float('-inf')))
        #breakpoint()
        #energy1 = torch.dot(energy, tgt_mask)
        
        attention = self.softmax(energy1) # BX (N) X (N) 
        proj_value = self.value_conv(x) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        # out = out.view(m_batchsize,C,width,height)
        
        # out = self.gamma*out + x
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


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=4, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(2, conv_dim, 4, 2, 1)))
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

        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.project = nn.Conv1d(256, 80, kernel_size=1)

        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.drop3 = nn.Dropout(p=0.5)

        self.batchnorm1 = nn.BatchNorm1d(80)
        self.batchnorm2 = nn.BatchNorm1d(80)

        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x, enc_output):
        # noise1 = (torch.rand(x.shape) - 0.5) / 0.5
        # noise2 = (torch.rand(x.shape) - 0.5) / 0.5

        # noise1 = noise1.to(x.device)
        # noise2 = noise2.to(x.device)
        #breakpoint()

        enc_output = enc_output.permute(0, 2, 1)
        enc_output = self.upsampling(enc_output)
        enc_output = self.project(enc_output)
        enc_output = self.relu(enc_output)
        
        enc_output = self.batchnorm1(enc_output)
        x = self.batchnorm2(x)

        # enc_output += noise1
        # x += noise2

        enc_output = enc_output.unsqueeze(1)

        x = x.unsqueeze(1)


        x = torch.cat((x, enc_output), dim=1)
        out = self.l1(x)
        #out = self.drop3(out)

        out = self.l2(out)
        out = self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)
        
        #(写真ごとに確率算出)
        out = torch.mean(out, dim=(2, 3))
        return out.squeeze(), p1, p2


class Discriminator_suvey(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=4, image_size=64, conv_dim=512):
        super(Discriminator_suvey, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv1d(80+256, 512, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv1d(curr_dim, 1, 1))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn_suvey(2048, 'relu')
        self.attn2 = Self_Attn_suvey(4096, 'relu')

        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.project = nn.Conv1d(256, 80, kernel_size=1)

        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.drop3 = nn.Dropout(p=0.5)

        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(80)

        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x, enc_output):
        
        noise2 = (torch.rand(x.shape) - 0.5) / 0.5
        noise2 = noise2.to(x.device)
        #breakpoint()

        enc_output = enc_output.permute(0, 2, 1)
        enc_output = self.upsampling(enc_output)

        #noise2 = (torch.rand(enc_output.shape) - 0.5) / 0.5

        # enc_output = self.project(enc_output)
        # enc_output = self.relu(enc_output)
        noise1 = (torch.rand(enc_output.shape) - 0.5) / 0.5
        noise1 = noise1.to(x.device)
        
        enc_output = self.batchnorm1(enc_output)
        x = self.batchnorm2(x)

        enc_output += noise1
        x += noise2

        x = torch.cat((x, enc_output), dim=1)

        #breakpoint()
        # enc_output = enc_output.unsqueeze(1)

        # x = x.unsqueeze(1)


        # x = torch.cat((x, enc_output), dim=1)

        
        out = self.l1(x)
        #breakpoint()
        out = self.drop3(out)

        out = self.l2(out)
        out = self.l3(out)
        #breakpoint()
        out,p1 = self.attn1(out)

        out=self.l4(out)
        out.shape
        out,p2 = self.attn2(out)
        out=self.last(out)
        
        #(写真ごとに確率算出)
        out = torch.mean(out, dim=(2))

        return out.squeeze(), p1, p2

class Discriminator_suvey_test(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=4, image_size=64, conv_dim=512, PL=False):
        super().__init__()
        self.imsize = image_size
        self.PL = PL

        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv1d(80+256, 512, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if PL:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv1d(curr_dim, 1, 1))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn_suvey(2048, 'relu')
        self.attn2 = Self_Attn_suvey(2048, 'relu')

        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.project = nn.Conv1d(256, 80, kernel_size=1)

        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(80)

        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x, enc_output):
        
        noise2 = (torch.rand(x.shape) - 0.5)
        noise2 = noise2.to(x.device)
        #breakpoint()

        enc_output = enc_output.permute(0, 2, 1)
        enc_output = self.upsampling(enc_output)

        #noise2 = (torch.rand(enc_output.shape) - 0.5) / 0.5

        # enc_output = self.project(enc_output)
        # enc_output = self.relu(enc_output)
        noise1 = (torch.rand(enc_output.shape) - 0.5)
        noise1 = noise1.to(x.device)
        
        enc_output = self.batchnorm1(enc_output)
        x = self.batchnorm2(x)

        enc_output += noise1
        x += noise2

        x = torch.cat((x, enc_output), dim=1)

        #breakpoint()
        # enc_output = enc_output.unsqueeze(1)

        # x = x.unsqueeze(1)


        # x = torch.cat((x, enc_output), dim=1)

        
        out = self.l1(x)
        #breakpoint()
        out = self.drop3(out)

        out = self.l2(out)
        out = self.l3(out)
        #breakpoint()
        out,p1 = self.attn1(out)

        if self.PL:
            out=self.l4(out)

        out,p2 = self.attn2(out)
        out=self.last(out)
        
        #(写真ごとに確率算出)
        out = torch.mean(out, dim=(2))

        return out.squeeze(), p1, p2


class Discriminator_suvey_test2(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=4, image_size=64, conv_dim=512, PL=False):
        super().__init__()
        self.imsize = image_size
        self.PL = PL

        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv1d(80+256, 512, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if PL:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv1d(curr_dim, 1, 1))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn_suvey(2048, 'relu')
        if self.PL:
            self.attn2 = Self_Attn_suvey(4096, 'relu')
        else:
            self.attn2 = Self_Attn_suvey(2048, 'relu')

        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.project = nn.Conv1d(256, 80, kernel_size=1)

        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.drop3 = nn.Dropout(p=0.5)

        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(80)

        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x, enc_output):
        
        # noise2 = (torch.rand(x.shape) - 0.5)
        # noise2 = noise2.to(x.device)
        #breakpoint()

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

        #breakpoint()
        # enc_output = enc_output.unsqueeze(1)

        # x = x.unsqueeze(1)


        # x = torch.cat((x, enc_output), dim=1)

        
        out = self.l1(x)
        #breakpoint()
        out = self.drop3(out)

        out = self.l2(out)
        out = self.l3(out)
        #breakpoint()
        out,p1 = self.attn1(out)

        if self.PL:
            out=self.l4(out)

        out,p2 = self.attn2(out)
        out=self.last(out)
        
        #(写真ごとに確率算出)
        out = torch.mean(out, dim=(2))

        return out.squeeze(), p1, p2



class Discriminator_suvey_test3(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=4, image_size=64, conv_dim=512, PL=False):
        super().__init__()
        self.imsize = image_size
        self.PL = PL

        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv1d(80+256, 512, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if PL:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv1d(curr_dim, 1, 1))
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
        #breakpoint()
        data_mask = make_pad_mask(data_len, max_len)
        out_mask = make_mask_out(data_len, max_len)

        enc_output = enc_output.permute(0, 2, 1)
        enc_output = self.upsampling(enc_output)

        #noise2 = (torch.rand(enc_output.shape) - 0.5) / 0.5

        # enc_output = self.project(enc_output)
        # enc_output = self.relu(enc_output)
        # noise1 = (torch.rand(enc_output.shape) - 0.5)
        # noise1 = noise1.to(x.device)
 
        #enc_output = self.batchnorm1(enc_output)
        #x = self.batchnorm2(x)
       
        # enc_output += noise1
        # x += noise2
       
        x = torch.cat((x, enc_output), dim=1)

        # enc_output = enc_output.unsqueeze(1)

        # x = x.unsqueeze(1)


        # x = torch.cat((x, enc_output), dim=1)
        # max_len = torch.tensor(10)
        # data = [3, 4, 5, 10]

        # data = torch.tensor(data)
        mask = make_pad_mask(data, max_len, data_mask)
        
        out = self.l1(x)
    
        out = self.drop3(out)

        out = self.l2(out)
        out = self.l3(out)
        
        out,p1 = self.attn1(out, mask)

        if self.PL:
            out=self.l4(out)

        out,p2 = self.attn2(out, mask)
        out=self.last(out)
        
        #(写真ごとに確率算出)
        out1 = out.masked_fill(mask_out, torch.tensor(0))
        out1 = torch.mean(out1, dim=(2))

        return out1.squeeze(), p1, p2

class Discriminator_simple(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=4, image_size=64, conv_dim=64):
        super(Discriminator_simple, self).__init__()

        print('SAGAN simple')
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
        #breakpoint()
        x = x.unsqueeze(1)

        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)
        
        #(写真ごとに確率算出)
        out = torch.mean(out, dim=(2, 3))
        return out.squeeze(), p1, p2


class Discriminator_suvey_test2(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=4, image_size=64, conv_dim=512, PL=False):
        super().__init__()
        self.imsize = image_size
        self.PL = PL

        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv1d(80+256, 512, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if PL:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv1d(curr_dim, curr_dim * 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv1d(curr_dim, 1, 1))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn_suvey(2048, 'relu')
        if self.PL:
            self.attn2 = Self_Attn_suvey(4096, 'relu')
        else:
            self.attn2 = Self_Attn_suvey(2048, 'relu')

        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.project = nn.Conv1d(256, 80, kernel_size=1)

        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.drop3 = nn.Dropout(p=0.5)

        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(80)

        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x, enc_output):
        
        # noise2 = (torch.rand(x.shape) - 0.5)
        # noise2 = noise2.to(x.device)
        #breakpoint()

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

        # enc_output = enc_output.unsqueeze(1)

        # x = x.unsqueeze(1)


        # x = torch.cat((x, enc_output), dim=1)
        max_len = torch.tensor(10)
        data = [3, 4, 5, 10]

        data = torch.tensor(data)
        mask = make_pad_mask(data, max_len)
        mask_out = make_mask_out(data, max_len)
        
        out = self.l1(x)
    
        out = self.drop3(out)

        out = self.l2(out)
        out = self.l3(out)
        
        out,p1 = self.attn1(out, mask)

        if self.PL:
            out=self.l4(out)

        out,p2 = self.attn2(out, mask)
        out=self.last(out)
        
        #(写真ごとに確率算出)
        out1 = out.masked_fill(mask_out, torch.tensor(0))
        out1 = torch.mean(out1, dim=(2))

        return out1.squeeze(), p1, p2



def train_SAGAN_one_epoch(G, D, opt_D, opt_G, train_loader, training_method, mixing_prob, loss_f, device, change_g, change_d, epoch):
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
        g_out_fake, _, _ = D(output, enc_output)
        g_out_real, _, _ = D(feature, enc_output_tmp)

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

        d_out_real, _, _ = D(feature, enc_output_tmp2)
        d_out_fake, _, _ = D(output, enc_output_tmp3)
      
        d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
        d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

        loss_D = d_loss_fake + d_loss_real

        for i in range(d_out_real.shape[0]):
            prob = d_out_real[0]

            if prob>=0:
                correct_cnt += 1
            else:
                wrong_cnt += 1

            batch_all += 1

        for i in range(d_out_fake.shape[0]):
            prob = d_out_fake[0]

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

    if correct_cnt>=0.4:
        change_g = True
    else:
        change_g = False

    if correct_cnt<0.65:
        change_d = True
    else:
        change_d = False

    return output_loss_all, dec_output_loss_all, loss_G_all, loss_D_all, correct_cnt, wrong_cnt, change_g, change_d

def train_SAGAN_one_simple_epoch(G, D, opt_D, opt_G, train_loader, training_method, mixing_prob, loss_f, device):
    iter_cnt = 0
    all_iter = len(train_loader)
    loss_G_all = 0
    loss_D_all = 0
    output_loss_all = 0
    dec_output_loss_all = 0

    print('start simple GAN')

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        iter_cnt += 1

        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len, speaker = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device), speaker.to(device)
        output, dec_output, feat_add_out = G(lip=lip, prev=feature, data_len=data_len, training_method=training_method, mixing_prob=mixing_prob)               

        output_tmp = output.detach()

        # train G(-logit + MSE loss)
        B, C, T = output.shape
        g_out_fake, _, _ = D(output)
        g_loss_fake = - g_out_fake.mean()

        #breakpoint()
        dec_output_loss = loss_f.mse_loss(dec_output, feature, data_len, max_len=T)
        output_loss = loss_f.mse_loss(output, feature, data_len, max_len=T)
        
        g_loss = 0.01*g_loss_fake + dec_output_loss + output_loss
        opt_D.zero_grad()
        opt_G.zero_grad()
        g_loss.backward()

        clip_grad_norm_(G.parameters(), 3.0)
        opt_G.step()
        
        output_loss_all += output_loss.item()
        dec_output_loss_all += dec_output_loss.item()
        loss_G_all += g_loss.item()
        del g_loss, output_loss, dec_output_loss
        torch.cuda.empty_cache()


        #train D()
        d_out_real, _, _ = D(feature)
      
        d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

        d_out_fake, _ , _ = D(output_tmp)
        d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

        loss_D = d_loss_fake + d_loss_real 
        opt_G.zero_grad()
        opt_D.zero_grad()
        loss_D.backward()
        clip_grad_norm_(D.parameters(), 3.0)
        opt_D.step()


        loss_D_all += loss_D.item()

        del loss_D, d_loss_fake, d_loss_real, output_tmp, feature
        torch.cuda.empty_cache()

        # if iter_cnt > 10:
        #     break
    output_loss_all /= all_iter
    dec_output_loss_all /= all_iter
    loss_G_all /= all_iter
    loss_D_all /= all_iter

    return output_loss_all, dec_output_loss_all, loss_G_all, loss_D_all

def train_SAGAN_PF_epoch(G, D, opt_D, opt_G, train_loader, training_method, mixing_prob, loss_f, device, change_g, change_d, epoch):
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
        g_out_real, _, _ = D(output, enc_output)
        g_out_fake, _, _ = D(FR_output, enc_output_tmp3)

        g_loss_real = g_out_real.mean()
        g_loss_fake = g_out_fake.mean()

        dec_output_loss = loss_f.mse_loss(dec_output, feature, data_len, max_len=T)
        output_loss = loss_f.mse_loss(output, feature, data_len, max_len=T)
        
        if change_g:
            #g_loss = -0.005*(g_loss_fake - g_loss_real) + dec_output_loss + output_loss
            g_loss = -0.0001*g_loss_fake + output_loss + dec_output_loss
        else:
            g_loss = output_loss + dec_output_loss


        output_loss_all += output_loss.item()
        dec_output_loss_all += dec_output_loss.item()
        loss_G_all += g_loss_fake.item()


        opt_D.zero_grad()
        opt_G.zero_grad()
        g_loss.backward()
        clip_grad_norm_(G.parameters(), 3.0)
        opt_G.step()

        del g_loss, output_loss, dec_output_loss, lip, output, enc_output, enc_output_tmp3, FR_output, g_loss_real, g_loss_fake
        torch.cuda.empty_cache()

        #train D()

        FR_output, _, enc_output = G(lip_tmp)
        d_out_real, _, _ = D(output_tmp, enc_output_tmp)
    
        d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

        d_out_fake, _ , _ = D(FR_output, enc_output_tmp2)
        d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

        for i in range(d_out_real.shape[0]):
            prob = d_out_real[0]

            if prob>=0:
                correct_cnt += 1
            else:
                wrong_cnt += 1

            batch_all += 1

        for i in range(d_out_fake.shape[0]):
            prob = d_out_fake[0]

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
            clip_grad_norm_(D.parameters(), 1.0)
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

    if correct_cnt>=0.4:
        change_g = True
    else:
        change_g = False

    if correct_cnt<0.65:
        change_d = True
    else:
        change_d = False

    return output_loss_all, dec_output_loss_all, loss_G_all, loss_D_all, correct_cnt, wrong_cnt, change_g, change_d




def train_SAGAN_PFsimple_epoch(G, D, opt_D, opt_G, train_loader, training_method, mixing_prob, loss_f, device, change_g, change_d):
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
    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        
        iter_cnt += 1
        
        lip, feature, feat_add, upsample, data_len, speaker, label = batch

        lip, feature, data_len = lip.to(device), feature.to(device), data_len.to(device)
        lip_tmp = lip.clone().detach()

        output, dec_output, enc_output = G(lip=lip, prev=feature, data_len=data_len, training_method=training_method, mixing_prob=mixing_prob) 
        output_tmp = output.clone().detach()
        # FR_output, _, RF_enc_output = G(lip=lip_tmp)   
        
        # train G(-logit + MSE loss) fake
        B, C, T = output.shape
        g_out_fake, _, _ = D(output)
        # for i in range(g_out_fake.shape[0]):
        #     prob = g_out_fake[i]
        #     if prob >= 0:
        #         correct_cnt += 1
        #     else:
        #         wrong_cnt += 1
        #     batch_all += 1
        
        g_loss_fake = - g_out_fake.mean()

        dec_output_loss = loss_f.mse_loss(dec_output, feature, data_len, max_len=T)
        output_loss = loss_f.mse_loss(output, feature, data_len, max_len=T)
        
        if change_g:
            g_loss = 0.0005*g_loss_fake + dec_output_loss + output_loss
        else:
            g_loss = output_loss + dec_output_loss


        output_loss_all += output_loss.item()
        dec_output_loss_all += dec_output_loss.item()
        loss_G_all += g_loss_fake.item()

       
        opt_D.zero_grad()
        opt_G.zero_grad()
        g_loss.backward()
        clip_grad_norm_(G.parameters(), 3.0)
        opt_G.step()

        del g_loss, output_loss, dec_output_loss, output, dec_output
        torch.cuda.empty_cache()

        #train D()

        FR_output, _, _ = G(lip_tmp)
        d_out_real, _, _ = D(output_tmp)
        d_out_fake, _ , _ = D(FR_output)

        for i in range(d_out_real.shape[0]):
            prob = d_out_real[0]

            if prob>=0:
                correct_cnt += 1
            else:
                wrong_cnt += 1

            batch_all += 1

        for i in range(d_out_fake.shape[0]):
            prob = d_out_fake[0]

            if prob<0:
                correct_cnt += 1
            else:
                wrong_cnt += 1

            batch_all += 1
    
        d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

    
        d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

        loss_D = d_loss_fake + d_loss_real 
        opt_G.zero_grad()
        opt_D.zero_grad()
        #print(f'loss D: {loss_D}')

        # if iter_cnt==1:
        #     breakpoint()
        if change_d:
            loss_D.backward()
            clip_grad_norm_(D.parameters(), 3.0)
            opt_D.step()
        
        loss_D_all += loss_D.item()

        del loss_D, FR_output, d_loss_fake, d_loss_real, lip_tmp, output_tmp
        torch.cuda.empty_cache()


        # if iter_cnt > 10:
        #     break

    

    output_loss_all /= all_iter
    dec_output_loss_all /= all_iter
    loss_G_all /= all_iter
    loss_D_all /= all_iter

    correct_cnt /= batch_all
    wrong_cnt /= batch_all

    
    if correct_cnt>=0.4:
        change_g = True
    else:
        change_g = False

    if correct_cnt<0.65:
        change_d = True
    else:
        change_d = False

    return output_loss_all, dec_output_loss_all, loss_G_all, loss_D_all, correct_cnt, wrong_cnt, change_g, change_d
# def train_G(G, D, opt_D, opt_G, output, dec_output, feature, )


# def train_G(G, D, opt_G, opt_D, lip, )

if __name__=='__main__':
  
    D = Discriminator()
    x = torch.randn(4, 1, 80, 300)
    y, _ ,_ = D(x)
    breakpoint()
