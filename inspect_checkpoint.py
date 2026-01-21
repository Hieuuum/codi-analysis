import torch
import sys

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "gpt2_commonsense/gsm8k_llama1b_latent_baseline/gpt2/ep_50/lr_0.003/seed_11/pytorch_model.bin"

print(f"Loading checkpoint from: {ckpt_path}")
state_dict = torch.load(ckpt_path, map_location='cpu')

print(f"\nTotal number of parameters: {len(state_dict.keys())}")
print("\nFirst 30 keys in the checkpoint:")
for i, key in enumerate(list(state_dict.keys())[:30]):
    param = state_dict[key]
    if hasattr(param, 'shape'):
        print(f"  {i+1}. {key}: shape {param.shape}, dtype {param.dtype}")
    else:
        print(f"  {i+1}. {key}: {type(param)}")

print("\n\nKey patterns:")
keys_list = list(state_dict.keys())
if any('lora' in k.lower() for k in keys_list):
    print("  - Contains LoRA weights")
if any('prj' in k.lower() for k in keys_list):
    print("  - Contains projection layer weights")
if any('codi' in k.lower() for k in keys_list):
    print("  - Contains CODI-specific weights")

# Count parameter types
lora_count = sum(1 for k in keys_list if 'lora' in k.lower())
prj_count = sum(1 for k in keys_list if 'prj' in k.lower())
print(f"\n  - LoRA parameters: {lora_count}")
print(f"  - Projection parameters: {prj_count}")
