import torch

p = r"D:\rsna_work\public_weights\rsna-knee-r6-fe055406\E_lrprior_s0_f0.pt"
sd = torch.load(p, map_location="cpu", weights_only=True)
print(f"Total keys: {len(sd)}")
for k in list(sd.keys())[:20]:
    print(k, sd[k].shape)
print("...")
for k in list(sd.keys())[-15:]:
    print(k, sd[k].shape)
