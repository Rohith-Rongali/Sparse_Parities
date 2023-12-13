import torch
from models import MLP
from utils import generate_data,train
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore

import logging
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    hydra.utils.log.info("Logging from main")
    print("Running main function...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    # alphas = [0.0,0.05,0.1,0.15,0.2]
    data_config = cfg.data
    model_config = cfg.model
    train_config = cfg.train

    data, labels = generate_data(data_config)
    labels=labels.unsqueeze(1)
    # Perform train-test split
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

    # Create dataloaders
    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    test_data = torch.utils.data.TensorDataset(x_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_config.batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model = MLP(model_config.hidden_layers, model_config.input_dim, model_config.output_dim, model_config.use_batchnorm, model_config.dropout_rate)
    model.to(device)
    # Create optimizer
    if train_config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=train_config.lr)
    elif train_config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)

    # Create loss function
    if train_config.loss_fn == 'MSE':
        loss_fn = torch.nn.MSELoss()

    train_loss,test_loss = train(model,optimizer,loss_fn,train_config.num_epochs,train_loader,(x_train, x_test, y_train, y_test),device)

    # Plot train and test losses
    plt.plot(range(len(train_loss)), train_loss, label=f"Train Loss, n={data_config.n}, k={data_config.k}, alpha={data_config.alpha}")
    plt.plot(range(len(test_loss)), test_loss, label=f"Test Loss, n={data_config.n}, k={data_config.k}, alpha={data_config.alpha}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    #save img in outputs folder
    output_path = str(data_config.num_points)+'_'+str(train_config.num_epochs)+"_loss_plot.png"
    plt.savefig(output_path)


    # test_losses = []

    # for alpha in alphas:
    #     # Create model
    #     # Generate data and labels
    #     data, labels = generate_data(num_points, n, k, alpha, bias)

    #     # Perform train-test split
    #     x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

    #     # Create dataloaders
    #     train_data = torch.utils.data.TensorDataset(x_train, y_train)
    #     test_data = torch.utils.data.TensorDataset(x_test, y_test)
    #     train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    #     test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    #     model = MLP(hidden_layers, input_dim, output_dim, use_batchnorm, dropout_rate)
    #     model.to(device)
    #     # Create optimizer
    #     optimizer = torch.optim.Adam(model.parameters(), lr=lr)     

    #     # Train model
    #     train_loss, test_loss = train(model, num_epochs)
    #     test_losses.append(test_loss)  # Store the test_loss values

    # # Plot test_losses
    # for i, test_loss in enumerate(test_losses):
    #     plt.plot(range(len(test_loss)), test_loss, label=f"Alpha: {alphas[i]}")

    # plt.xlabel("Epoch")
    # plt.ylabel("Test Loss")
    # plt.legend()

    # # Save the image in the "outputs" folder
    # output_folder = "outputs"
    # os.makedirs(output_folder, exist_ok=True)
    # output_path = os.path.join(output_folder, "test_loss_plot.png")
    # plt.savefig(output_path)


if __name__ == "__main__":
    main()









