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


# configuration file 
config_file_flower = "pyproject.toml"
def parse_config_file(config_file):
    num_supernodes = 0
    with open(config_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("batch-size"):
            batch_size = int(line.split("=")[1].strip())
        elif line.startswith("learning-rate"):
            learning_rate = float(line.split("=")[1].strip())
        elif line.startswith("local-epochs"):
            epochs = int(line.split("=")[1].strip())
        elif line.startswith("options.num-supernodes"):
            if num_supernodes == 0:
                num_supernodes = int(line.split("=")[1].strip())
    return batch_size, learning_rate, epochs, num_supernodes
batch_size, lr, epochs, num_partitions = parse_config_file(config_file_flower)

print(f"Batch size: {batch_size}")
print(f"Learning rate: {lr}")
print(f"Local epochs: {epochs}")
print(f"Number of supernodes: {num_partitions}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# Initialize model and load data
model = Net()
train_dataset, time_steps_extrap = get_dataset(dataset_name = dataset_name, type="train")
test_dataset, _ = get_dataset(dataset_name = dataset_name, type="test")

# train dataset 

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=False,
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

