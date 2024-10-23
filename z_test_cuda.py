import torch

print(f"is Cuda available? \n{torch.cuda.is_available()}")
print(f"the number of divices: {torch.cuda.device_count()}")
print(f"The current device : {torch.cuda.current_device()}")
print(f"Current default device: {torch.cuda.device(0)}")
print(f"Cuda : {torch.cuda.get_device_name(0)}")

import torch.cuda

print("Done")