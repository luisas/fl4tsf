import torch
from torch.utils.data import DataLoader
from task import Net, load_data, train, test
import pandas as pd
import os
from pytorch_example.get_dataset import get_dataset, basic_collate_fn


# Hyperparameters
dataset_name = "periodic"
sample_tp = 0.5
cut_tp = None
extrap = False
batch_size = 50
epochs = 300
lr = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize model and load data
model = Net()
train_dataset, time_steps_extrap = get_dataset(dataset_name = dataset_name, type="train")
# check how big the dataset is 
print(f"Train dataset size: {len(train_dataset)}")
test_dataset, _ = get_dataset(dataset_name = dataset_name, type="test")

# train dataset 

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True,
    collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, dataset_name, sample_tp, cut_tp, extrap, data_type = "train"))
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False,
    collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, dataset_name, sample_tp, cut_tp, extrap, data_type = "test"))

# train
loss_training = train(model, train_loader, epochs, lr=lr, device=device, loss_per_epoch=True)

# test
loss, accuracy = test(Net(), train_loader, device)
print(f"Test loss before training: {loss:.4f}")
print(f"Test accuracy before training: {accuracy:.4f}")

loss, accuracy = test(model, test_loader, device)
print(f"Test loss after training: {loss:.4f}")
print(f"Test accuracy after training: {accuracy:.4f}")

#######################################
# Store files
#######################################

df = pd.DataFrame(loss_training)
if not os.path.exists("outputs/centralized_outputs"):
    os.makedirs("outputs/centralized_outputs", exist_ok=True)
# add epochs and header
df.columns = ["loss"]
df.index.name = "epoch"
df.to_csv("outputs/centralized_outputs/loss_per_epoch.csv", index=True)
df_test = pd.DataFrame({"loss": loss, "accuracy": accuracy}, index=[0])
df_test.to_csv("outputs/centralized_outputs/test_loss_accuracy.csv", index=False)

