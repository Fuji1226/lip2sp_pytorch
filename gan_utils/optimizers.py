import torch


def make_optimizer(model_g, model_d, lr, beta1, beta2, weight_decay):
    optimizer_g = torch.optim.Adam(
            params=model_g.parameters(),
            lr=lr, 
            betas=(beta1, beta2),
            weight_decay=weight_decay    
        )

    optimizer_d = torch.optim.Adam(
        params=model_d.parameters(),
        lr=lr, 
        betas=(beta1, beta2),
        weight_decay=weight_decay    
    )
    
    return optimizer_g, optimizer_d