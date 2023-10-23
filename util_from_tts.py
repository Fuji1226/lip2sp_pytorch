import torch

def load_from_tts(model, ckpt_path):
    checkpoint = torch.load(str(ckpt_path))['model']
    
    save_module = ['decoder']
    partial_state_dict = {}
    for key, value in checkpoint.items():
       for module in save_module:
           if module in key:
               key = key.replace(f'{module}.', '')
               partial_state_dict[key] = value
               break
           
    model.decoder.load_state_dict(partial_state_dict)

    save_module = ['postnet']
    partial_state_dict = {}
    for key, value in checkpoint.items():
       for module in save_module:
           if module in key:
               key = key.replace(f'{module}.', '')
               partial_state_dict[key] = value
               break
           
    model.postnet.load_state_dict(partial_state_dict)
    
    #model固定
    for param in model.decoder.parameters():
        param.requires_grad = False
    for param in model.postnet.parameters():
        param.requires_grad = False
    return model

def model_grad_ok(model):
    for param in model.decoder.parameters():
        param.requires_grad = True
    for param in model.postnet.parameters():
        param.requires_grad = True
    return model