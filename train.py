
from data import getdata
from model import NequIP, force
import os

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from trainutils import loadmodel, initialize_shift_scale, savecheckpoint
from test import evaluate


# from evaluate import peratom_mse



def criterion(energy, forces, data):
    
    losse = F.mse_loss(energy, data.y)

    lossf = F.mse_loss(forces, data.forces)

    we = 1.0
    wf = 100.0

    losstot = we * losse + wf * lossf
    # print(f"Energy Loss: {losse.item():.6f}, Force Loss: {lossf.item():.6f}, Total Loss: {losstot.item():.6f}")

    return losstot

def train(finetune=False, batch_size=32, checkpoint_ft='model_E0.pth'):
    import time

    trainloader, valloader, _ = getdata(mini=False, batch_size=batch_size)
    trainsize = int(len(trainloader.dataset) / batch_size)
    print('Data loaded')
    
    basedir = "/content/drive/My Drive/MS-Physics/ML-DFT/NequIP/"
    
    checkpoints_dir = os.path.join(basedir, "checkpoints")
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    log_dir = os.path.join(basedir, "runs")

    writer = SummaryWriter(log_dir=log_dir)

    if finetune:
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_ft)
        model = loadmodel(checkpoint_path)
        print(f"Loaded model from {checkpoint_path} for finetuning.")
    else:
        model = NequIP()
        print("Initialized new model for training.")

    # --- PLACE THIS BEFORE YOUR TRAINING LOOP ---
    initialize_shift_scale(model, trainloader)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    epochs = 5000

    model.train()
    
    print("Starting training...")
    for epoch in range(epochs+1):

        stime = time.time()
        for k, batch in enumerate(trainloader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pos = batch.pos.requires_grad_(True)
            
            # Forward pass with the model
            energy = model(batch.z, pos, batch.batch)

            forces = force(energy, pos)
            
            loss = criterion(energy, forces, batch)

            loss.backward()
            optimizer.step()

            writer.add_scalar('Batch-Loss/train', loss.item(), trainsize*epoch + k)
            
        # --- save model every 10 epochs ---
        if epoch % 1 == 0:
            # savefig(predictions, targets, epoch)
            savecheckpoint(epoch, model, optimizer, loss)

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Time Taken: {(time.time()-stime): .01f}")


            mae_energy, mae_force = evaluate(model, valloader, device=device)
            writer.add_scalar('MAE-Energy/val', mae_energy, epoch)
            writer.add_scalar('MAE-Force/val', mae_force, epoch)
            

    writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train NequIP model.')
    parser.add_argument('--finetune', default=False, type=bool, help='Fine-tune from a ' \
    'pre-trained model.')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training.')
    parser.add_argument('--checkpoint_ft', default="model_E0.pt", type=str, help='Checkpoint to ' \
    'load from when finetuning')
    args = parser.parse_args()


    train(finetune=args.finetune, batch_size=args.batch_size, checkpoint_ft=args.checkpoint_ft)