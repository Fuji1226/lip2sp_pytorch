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
    return model


def load_from_vqvae(model, ckpt_path):
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

    save_module = ['vq']
    partial_state_dict = {}
    for key, value in checkpoint.items():
       for module in save_module:
           if module in key:
               key = key.replace(f'{module}.', '')
               partial_state_dict[key] = value
               break
           
    model.vq.load_state_dict(partial_state_dict)
    
    #model固定
    for param in model.decoder.parameters():
        param.requires_grad = False
    for param in model.vq.parameters():
        param.requires_grad = False
        
    return model

def load_from_vqvae_ctc(model, ckpt_path):
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

    save_module = ['vq']
    partial_state_dict = {}
    for key, value in checkpoint.items():
       for module in save_module:
           if module in key:
               key = key.replace(f'{module}.', '')
               partial_state_dict[key] = value
               break
           
    model.vq.load_state_dict(partial_state_dict)
    
    save_module = ['ctc_output_layer']
    partial_state_dict = {}
    for key, value in checkpoint.items():
       for module in save_module:
           if module in key:
               key = key.replace(f'{module}.', '')
               partial_state_dict[key] = value
               break
           
    model.ctc_output_layer.load_state_dict(partial_state_dict)
    
    #model固定
    for param in model.decoder.parameters():
        param.requires_grad = False
    for param in model.vq.parameters():
        param.requires_grad = False

    for param in model.ctc_output_layer.parameters():
        param.requires_grad = False
    
    return model


        
def load_from_vqvae_mlm(model, vq_path, mlm_path):
    checkpoint = torch.load(str(vq_path))['model']
    
    save_module = ['decoder']
    partial_state_dict = {}
    for key, value in checkpoint.items():
       for module in save_module:
           if module in key:
               key = key.replace(f'{module}.', '')
               partial_state_dict[key] = value
               break
           
    model.decoder.load_state_dict(partial_state_dict)

    save_module = ['vq']
    partial_state_dict = {}
    for key, value in checkpoint.items():
       for module in save_module:
           if module in key:
               key = key.replace(f'{module}.', '')
               partial_state_dict[key] = value
               break
           
    model.vq.load_state_dict(partial_state_dict, strict=False)
    
    checkpoint = torch.load(str(mlm_path))['model']
    save_module = ['encoder']
    partial_state_dict = {}
    for key, value in checkpoint.items():
       for module in save_module:
           if module in key:
               key = key.replace(f'{module}.', '')
               partial_state_dict[key] = value
               break
    
    
    #model固定
    for param in model.decoder.parameters():
        param.requires_grad = False
    for param in model.vq.parameters():
        param.requires_grad = False
    return model

def load_lmmodel(model, lm_path):
    checkpoint = torch.load(str(lm_path))['model']
    
    save_module = ['lm_decoder']
    partial_state_dict = {}
    
    for key, value in checkpoint.items():
       for module in save_module:
           if module in key:
               key = key.replace(f'{module}.', '')
               partial_state_dict[key] = value
               break
           
    model.lm_decoder.load_state_dict(partial_state_dict)

    save_module = ['ctc_output_layer']
    partial_state_dict = {}
    
    for key, value in checkpoint.items():
       for module in save_module:
           if module in key:
               key = key.replace(f'{module}.', '')
               partial_state_dict[key] = value
               break
           
    model.ctc_output_layer.load_state_dict(partial_state_dict)
    
    for param in model.lm_decoder.parameters():
        param.requires_grad = False
    for param in model.ctc_output_layer.parameters():
        param.requires_grad = False
    
    return model 
        
def model_grad_ok(model):
    for param in model.decoder.parameters():
        param.requires_grad = True
    for param in model.postnet.parameters():
        param.requires_grad = True
    return model