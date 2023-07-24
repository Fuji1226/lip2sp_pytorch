import torch
import torch.nn as nn
import torch.nn.functional as F


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)
    
    
class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
     
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        #random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class MyCLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super().__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim),
        )

        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim),
            nn.Tanh(),
        )

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
     
    def loglikeli(self, x_samples, y_samples):
        """
        x_samples(speaker embedding) : (B, C)
        y_samples(vq) : (B, T, C)
        """
        mu, logvar = self.get_mu_logvar(x_samples)  # (B, C)
        mu = mu.unsqueeze(1).expand(-1, y_samples.shape[1], -1).reshape(-1, mu.shape[-1])   # (B, 1, C) -> (B, T, C) -> (B * T, C)
        logvar = logvar.unsqueeze(1).expand(-1, y_samples.shape[1], -1).reshape(-1, logvar.shape[-1])   # (B, 1, C) -> (B, T, C) -> (B * T, C)
        y_samples = y_samples.reshape(-1, y_samples.shape[-1])  # (B, T, C) -> (B * T, C)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def forward(self, x_samples, y_samples):
        """
        x_samples(speaker embedding) : (B, C)
        y_samples(vq) : (B, T, C)
        """
        mu, logvar = self.get_mu_logvar(x_samples)  # (B, C)
        mu = mu.unsqueeze(1).expand(-1, y_samples.shape[1], -1)   # (B, 1, C) -> (B, T, C)
        
        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()
        
        # positive = - (mu - y_samples)**2 / logvar.exp()
        # negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        positive = - ((mu - y_samples)**2).mean(dim=1) / logvar.exp()
        negative = - ((mu - y_samples[random_index])**2).mean(dim=1) / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


if __name__ == "__main__":
    bs = 16
    spk_emb = torch.rand(bs, 256)
    vq_out = torch.rand(bs, 64, 150)
    net = MyCLUBSample(x_dim=spk_emb.shape[1], y_dim=vq_out.shape[1], hidden_size=512)

    vq_out = vq_out.permute(0, 2, 1)
    loglikeli = net.loglikeli(spk_emb, vq_out)
    mutual_info = net(spk_emb, vq_out)
    print(loglikeli)
    print(mutual_info)