import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from unet import SSM5311
from utils import * 
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from unet import SSM5311
from dataset import ME5311Dataset
from utils import * 
from distutils.util import strtobool
from config import *

parser = argparse.ArgumentParser(description="train the shear prediction model")
txt2bool = lambda x:bool(strtobool(x))

parser.add_argument("--seed_value",     default=1000, type=int, nargs='?', help="training seeds")
parser.add_argument("--lr",             default=1e-3, type=float, nargs='?', help="learning rate")
parser.add_argument("--weight_decay",   default=1e-4, type=float, nargs='?', help="weight decay for the Adam optimizer")
parser.add_argument("--bs",             default=128, type=int, nargs='?', help="batch size")
parser.add_argument("--epoch",          default=50, type=int, nargs='?', help="total epoch number")
parser.add_argument("--data_type",      default="addthis", type=str, nargs='?', help="total epoch number")
parser.add_argument("--train_on_err",      default=True, type=txt2bool, nargs='?', help="total epoch number")

args = parser.parse_args()

# model parameters
data_type = args.data_type
train_on_err = args.train_on_err
err_str = "_err" if train_on_err else ""

# training parameters
seed_value = args.seed_value
lr = args.lr
weight_decay = args.weight_decay
bs = args.bs
epoch = args.epoch

notes = ""
loss_fcn = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# save directory of tensorboard logs
writer_dir = f"logs/{data_type}{err_str}_models"
ds_path = "updated_processed_data_w_interpolation" # dataset path


def train_model(model, train_loader, test_loader, num_epochs, learning_rate):
    torch.cuda.empty_cache()
    # create the tensorboard log directory
    log_dir = os.path.join("train_logs", writer_dir)
    writer = SummaryWriter(log_dir=log_dir)
    print("training log will be saved to: ", log_dir)

    # create the directory for saving the model
    save_dir = f"saved_models/{data_type}{err_str}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("trained models will be saved to: ", save_dir)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(" ")
        with tqdm(train_loader, unit="batch") as t:
            for i, data in enumerate(t, 0):
                x, y = data

                x_new  = reshape_to_square(x)

                x_new, y = x_new.to(torch.float32).to(device), y.to(torch.float32).to(device)
                optimizer.zero_grad()
                outputs, mu, logvar = model(x_new)
                outputs = reshape_back(outputs)
                # print(outputs.min().cpu(), outputs.max().cpu(), logvar.max().cpu())

                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
                # loss_reg = outputs.mean() ** 2
                loss = loss_fcn(outputs, y)
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()

                # Print loss
                running_loss += loss.item() + 2 * kl_loss.mean()
                t.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / (i + 1):.4g}")

        # Log loss to tensorboard
        writer.add_scalar("Loss/train", running_loss / len(train_loader), epoch+1)
        
        
        # Validation loop
        model.eval()
        validation_loss = 0.0
        baseline_loss = 0.0
        with torch.no_grad():
            with tqdm(test_loader, unit="batch") as t:
                for i, data in enumerate(t, 0):
                    x, y = data
                    x = x.to(device)
                    x_new = reshape_to_square(x)
                    x_new, y = x_new.to(torch.float32).to(device), y.to(torch.float32).to(device)
                    

                    optimizer.zero_grad()
                    outputs = model(x_new)
                    outputs = reshape_back(outputs)
                    loss = loss_fcn(outputs, y)
                    loss_b = loss_fcn(x, y)
                    validation_loss += loss.item()
                    baseline_loss += loss_b.item()
                    t.set_description(f"Baseline loss: {baseline_loss / (i+1):.4g}, Loss: {validation_loss / (i + 1):.4g}")

            # Log validation loss to tensorboard
            mean_validation_loss = validation_loss / len(test_loader)
            writer.add_scalar("Loss/validation", validation_loss / len(test_loader), epoch+1)
            writer.add_scalar("Loss/val_baseline", baseline_loss / len(test_loader), epoch+1)
            writer.add_scalars('Curves', {'Loss/train': running_loss / len(train_loader), 
                                          'Loss/test': validation_loss / len(test_loader), 
                                          'Loss/baseline': baseline_loss / len(test_loader)}, epoch+1, 
            )

        file_path = os.path.join(save_dir, f"model_{epoch+1:02}_{mean_validation_loss:.4g}" + ".pth")
        torch.save(model.state_dict(), file_path)


    print("Training finished!")
    writer.close()

if __name__ == "__main__":
    torch.cuda.empty_cache()

    # processed_data_thick_tacniq_skip
    train_set = ME5311Dataset(train=True, type="slp", t_size=11, use_err=True)
    test_set =  ME5311Dataset(train=False, type="slp", t_size=11, use_err=True)

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=True)
    
    model = SSM5311(down_channels=global_down_chnl)
    print(f"Total param number: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    train_model(model, train_loader, test_loader, num_epochs=epoch, learning_rate=lr)