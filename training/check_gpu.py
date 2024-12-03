import torch

if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    print(f'GPU is count: {torch.cuda.device_count()}')
else:
    print("GPU is not available, using CPU.")
