import numpy as np


class RankSVM():
    def __init__(self, lmd=1e-5, n_iter=100000):
        self.lmd = lmd
        self.n_iter = n_iter
    
    def make_pair(self, x):
        index1 = np.random.randint(0, x.shape[0])
        index2 = np.random.randint(0, x.shape[0])
        x_pair = (x[index1], x[index2])
        return x_pair
    
    def make_pair_mix(self, x_natural, x_synth):
        index1 = np.random.randint(0, x_natural.shape[0])
        index2 = np.random.randint(0, x_synth.shape[0])
        x_pair = (x_natural[index1], x_synth[index2])
        return x_pair
        
    def fit(self, x_natural, x_synth):
        self.w = np.zeros(x_natural.shape[1])
        
        for i in range(self.n_iter):
            eta = 1.0 / (self.lmd * (i + 1))
            
            # x_natural
            x_natural_pair = self.make_pair(x_natural)
            x_natural_dif = x_natural_pair[0] - x_natural_pair[1]
            self.w *= (1 - eta * self.lmd)
            if (1 - np.dot(x_natural_dif, self.w)) > 0:
                self.w += (eta * x_natural_dif)
            
            # x_synth
            x_synth_pair = self.make_pair(x_synth)
            x_synth_dif = x_synth_pair[0] - x_synth_pair[1]
            self.w *= (1 - eta * self.lmd)
            if (1 - np.dot(x_synth_dif, self.w)) > 0:
                self.w += (eta * x_synth_dif)
                
            # x_mix
            x_mix_pair = self.make_pair_mix(x_natural, x_synth)
            x_mix_dif = x_mix_pair[0] - x_mix_pair[1]
            self.w *= (1 - eta * self.lmd)
            if np.dot(x_mix_dif, self.w) > 0:
                self.w -= (eta * x_mix_dif)
                
    def predict(self, x):
        return np.dot(x, self.w)