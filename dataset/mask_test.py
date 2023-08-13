import torch
from torch import nn
import random

def spatial_masking(lip):
        """
        空間領域におけるマスク
        lip : (T, C, H, W)
        """
        lip = lip.permute(3, 0, 1, 2)
        T, C, H, W = lip.shape
        lip_aug = lip.clone()
        print(f'lip aug: {lip_aug.shape}')
        
        #if self.cfg.train.which_spatial_mask == "has":
        if False:
            input_type = lip.dtype
            lip_aug = lip_aug.to(torch.float32)
            unfold = nn.Unfold(kernel_size=H // 4, stride=H // 4)
            fold = nn.Fold(output_size=(H, W), kernel_size=H // 4, stride=H // 4)

            lip_aug = unfold(lip_aug)
            breakpoint()
            # n_mask = torch.randint(0, self.cfg.train.n_spatial_mask, (1,))
            # mask_idx = [i for i in range(lip_aug.shape[-1])]
            # mask_idx = random.sample(mask_idx, n_mask)

            n_mask = 8
            mask_idx = [i for i in range(lip_aug.shape[-1])]
            mask_idx = mask_idx[:40] + [i for i in range(56, 64, 1)] + [40, 41, 46, 47] + [48, 49, 54, 55]      # all
            # mask_idx = [i for i in range(0, 8, 1)] + [8, 9, 14, 15] + [16, 23] + [24, 31]\
            #     + [32, 33, 38, 39] +[40, 41, 46, 47] + [48, 49, 54, 55] + [i for i in range(56, 64, 1)]   # outline
            print(f"mask_index = {mask_idx}")
            breakpoint()
            for i in mask_idx:
                lip_aug[..., i] = 0
            
            lip_aug = fold(lip_aug).to(input_type)
            breakpoint()
        else:
           
            do_or_through = random.randint(0, 1)
            do_or_through = 1
            if do_or_through == 1:
                breakpoint()
                mask = torch.zeros(H, W)
                # x_center = torch.randint(0, W, (1,))
                # y_center = torch.randint(0, H, (1,))
                x_center = torch.randint(8 // 2, W - 8 // 2, (1,))
                y_center = torch.randint(8 // 2, H - 8 // 2, (1,))
                x1 = torch.clamp(x_center - 8// 2, min=0, max=W)
                x2 = torch.clamp(x_center + 8 // 2, min=0, max=W)
                y1 = torch.clamp(y_center - 8 // 2, min=0, max=W)
                y2 = torch.clamp(y_center + 8 // 2, min=0, max=W)
                mask[y1:y2, x1:x2] = 1

                mask = mask.to(torch.bool)
                mask = mask.unsqueeze(0).unsqueeze(0).expand_as(lip_aug)   # (T, C, H, W)
                lip_aug = torch.where(mask, torch.zeros_like(lip_aug), lip_aug)

        return lip_aug



if __name__=='__main__':
    lip = torch.rand(4, 3, 16, 16)
    lip = torch.rand(3, 48, 48, 300)
    # unfold = nn.Unfold(kernel_size=4, stride=4)
    # y = unfold(lip)

    y = spatial_masking(lip)
    breakpoint()