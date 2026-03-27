import torch
from common import get_pdn_small, get_pdn_medium

model_size = 'small' 
out_channels = 384
weights_path = f'models/teacher_{model_size}.pth'


if model_size == 'small':
    teacher = get_pdn_small(out_channels)
else:
    teacher = get_pdn_medium(out_channels)

state_dict = torch.load(weights_path, map_location='cpu')
teacher.load_state_dict(state_dict)

teacher.eval()
print(f"Successfully loaded EfficientAD {model_size} teacher weights.")

params = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {params}")
