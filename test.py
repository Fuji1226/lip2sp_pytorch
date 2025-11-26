import torch
x = torch.randn(2, 2, 28)


x_double = x[:, :, :2 * (x.shape[2] // 14)]
x_base = x[:, :, 2 * (x.shape[2] // 14):6 * (x.shape[2] // 14)]
x_half = x[:, :, 6 * (x.shape[2] // 14):]

print(x)
print(x_double.shape,x_double)
breakpoint()
print(x_base.shape,x_base)
breakpoint()
print(x_half.shape,x_half)