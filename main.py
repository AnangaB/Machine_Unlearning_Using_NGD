import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import time
from alg1 import unlearn
from ngd import NGD
from utils import evaluate_model, get_test_and_train_data, get_trained_OLP_model

#get data loaders
train_loader, delete_loader, remaining_loader, test_loader = get_test_and_train_data(1000)

#OLP models below 
model = get_trained_OLP_model(train_loader)

start_time_remainder = time.time()
remainder_model  = get_trained_OLP_model(remaining_loader)
end_time_remainder = time.time()
remainder_model_time = end_time_remainder - start_time_remainder

theta_s_parameters = {name: param.clone().detach().requires_grad_(True) for name, param in model.named_parameters()}
print(f"Remainder model evaluation took {remainder_model_time:.4f} seconds.")


# NGD Optimizer setup
ngd_optimizer = NGD(
    params=model.parameters(),
    lr=0.05,
    momentum=0.9,
    weight_decay=5e-4,
    alpha=4,  # NGD specific
    update_period=4,
)

# Define input size and criterion
input_size = 28 * 28  # For MNIST or similar dataset
criterion = nn.CrossEntropyLoss()

start_time_unlearned = time.time()
# Unlearning process
unlearn(model, delete_loader, train_loader, ngd_optimizer, criterion,  len(train_loader.dataset), 1000, input_size,weight_decay=1e-4, noise_std=1e-2)
end_time_unlearned = time.time()
unlearned_model_time = end_time_unlearned - start_time_unlearned


# Evaluate unlearned model and remainder model
unlearned_model_accuracy = evaluate_model(model, test_loader, input_size)
remainder_model_accuracy = evaluate_model(remainder_model, test_loader, input_size)



# Print results
print(f"Unlearned model evaluation took {unlearned_model_time:.4f} seconds.")
print(f"Remainder model evaluation took {remainder_model_time:.4f} seconds.")


print(f"Test Accuracy after unlearning: {unlearned_model_accuracy:.4f}")
print(f"Test Accuracy of remainder model: {remainder_model_accuracy:.4f}")
