"""pytorch-example: A Flower / PyTorch app."""

import torch
from flower.task import Net, get_weights, load_data, set_weights, test, train

from flwr.client import ClientApp, NumPyClient
from flwr.common import Array, ArrayRecord, Context, RecordDict
import os
import json
from flower.model_config import get_model_config


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    """A simple client that showcases how to use the state.

    It implements a basic version of `personalization` by which
    the classification layer of the CNN is stored locally and used
    and updated during `fit()` and used during `evaluate()`.
    """

    def __init__(
        self, net, client_state: RecordDict, trainloader, valloader, local_epochs, num_client=None, round = 0 
    ):
        self.net: Net = net
        self.client_state = client_state
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.local_layer_name = "classification-head"
        self.num_client = num_client
        self.results = {}

    def fit(self, parameters, config):
        """Train model locally.

        The client stores in its context the parameters of the last layer in the model
        (i.e. the classification head). The classifier is saved at the end of the
        training and used the next time this client participates.
        """

        # Apply weights from global models (the whole model is replaced)
        set_weights(self.net, parameters)
        self.net.to(self.device)
        #config = get_model_config(file_path="model.config")
        # print round
        learning_rate = float(config["lr"])

        print("CONFIGURATION:")
        print (f"Learning rate: {learning_rate}")
        print(config)



        # Override weights in classification layer with those this client
        # had at the end of the last fit() round it participated in
        #self._load_layer_weights_from_state()


        train_loss, nodesolve, metric_dict = train(
            net = self.net,
            trainloader= self.trainloader,
            valloader = self.valloader,
            epochs = self.local_epochs,
            lr=learning_rate,
            device=self.device,
            loss_per_epoch=True,
        )
        epoch_loss = metric_dict["train_loss"]
        epoch_mse = metric_dict["train_mse"]
        val_loss = metric_dict["val_loss"]
        val_mse = metric_dict["val_mse"]
        nodesolves = metric_dict["nodesolves"]
        weights = metric_dict["weights"]
        grad_norms = metric_dict["grad_norms"]
        lr = metric_dict["lr"]

        self._store_results(
            tag="client_train",
            client=self.num_client,
            results_dict={"train_loss": epoch_loss, "train_mse": epoch_mse, "val_loss": val_loss, "val_mse": val_mse, "nodesolve": nodesolves, "weights": weights, "grad_norms": grad_norms, "lr": lr},
        )

        # Save classification head to context's state to use in a future fit() call
        # self._save_layer_weights_to_state()

        # Return locally-trained model and metrics
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss,  "nodesolve": nodesolve}
        )
    # def _save_layer_weights_to_state(self):
    #     """Save last layer weights to state."""
    #     arr_record = ArrayRecord(self.net.decoder.state_dict())

    #     # Add to RecordDict (replace if already exists)
    #     self.client_state[self.local_layer_name] = arr_record

    # def _load_layer_weights_from_state(self):
    #     """Load last layer weights to state."""
    #     if self.local_layer_name not in self.client_state.array_records:
    #         return

    #     state_dict = self.client_state[self.local_layer_name].to_torch_state_dict()

    #     # apply previously saved classification head by this client
    #     self.net.decoder.load_state_dict(state_dict, strict=True)


    def _store_results(self, tag: str, client, results_dict):
        """Store results in JSON file, with automatic round tracking."""
        # Ensure the directory exists
        output_dir = "federated_outputs"
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{output_dir}/results_{client}.json"

        # Load existing results from disk if file exists
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as fp:
                results = json.load(fp)
        else:
            results = {}

        # Compute the round number
        round_number = len(results.get(tag, []))
        results_dict["round"] = round_number

        # Update results
        if tag in results:
            results[tag].append(results_dict)
        else:
            results[tag] = [results_dict]

        # Save updated results to disk
        with open(filename, "w", encoding="utf-8") as fp:
            json.dump(results, fp, indent=2)


    def evaluate(self, parameters, config):
        """Evaluate the global model on the local validation set.

        Note the classification head is replaced with the weights this client had the
        last time it trained the model.
        """
        set_weights(self.net, parameters)
        # Override weights in classification layer with those this client
        # had at the end of the last fit() round it participated in
        # self._load_layer_weights_from_state()
        self.net.to(self.device)
        loss, accuracy = test(self.net, self.valloader, self.device, kl_coef = 0.993)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    model_config = get_model_config(file_path="model.config")
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = int(model_config["batch_size"])
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size)
    local_epochs = int(model_config["localepochs"])

    # Return Client instance
    # We pass the state to persist information across
    # participation rounds. Note that each client always
    # receives the same Context instance (it's a 1:1 mapping)
    client_state = context.state

    return FlowerClient(
        net, client_state, trainloader,valloader, local_epochs, partition_id
    ).to_client()



