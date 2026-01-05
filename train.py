
import os
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from data import getdata
from model import NequIP, force
from trainutils import loadmodel, initialize_shift_scale, savecheckpoint, count_parameters
from test import evaluate


def criterion(energy, forces, data):
    
    losse = F.mse_loss(energy, data.y)

    lossf = F.mse_loss(forces, data.forces)

    we = 1.0
    wf = 100.0

    losstot = we * losse + wf * lossf

    return losstot

def train(data_dir, results_dir, finetune, batch_size, checkpoint_ft, mps=False, kaggle=False,
          lr=1e-2):

    trainloader, valloader, _ = getdata(data_dir, mini=False, batch_size=batch_size)
    trainsize = int(len(trainloader.dataset) / batch_size)
    print('Data loaded')
    
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    writer = SummaryWriter()
    f = open('training-logs.txt', 'w')

    if kaggle:
        from kaggle_secrets import UserSecretsClient
        import wandb
        # --- 1. LOGIN SECURELY ---
        user_secrets = UserSecretsClient()
        secret_value = user_secrets.get_secret("wandb_key")
        wandb.login(key=secret_value)

        # --- 2. INITIALIZE RUN ---
        # This creates a "project" on your dashboard
        run = wandb.init(
            project="Thesis_NequIP_Aspirin",
            name="Run_01_1k_Samples", # Optional: Name this specific attempt
            config={
                "learning_rate": 0.005,
                "batch_size": 32,
                "max_epochs": 5000,
                "architecture": "NequIP",
                "dataset_size": 2000
            }
        )

    if finetune:
        if kaggle:
            restored_ckpt = wandb.restore('results/checkpoints/model_E990.pt', run_path="isamabdullah88-lahore-university-of-management-sciences/Thesis_NequIP_Aspirin/22bm6hom")
            checkpoint_path = restored_ckpt.name
        else:
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint_ft)
        
        model = loadmodel(checkpoint_path, args.mps)
        print(f"Loaded model from {checkpoint_path} for finetuning.")
    else:
        model = NequIP(mps=mps)
        print("Initialized new model for training.")
        # --- PLACE THIS BEFORE YOUR TRAINING LOOP ---
        model = initialize_shift_scale(model, trainloader)

    count_parameters(model)

    if mps:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

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
        if epoch % 10 == 0:
            
            checkpoint_path = os.path.join(checkpoints_dir, f"model_E{epoch}.pt")
            savecheckpoint(checkpoint_path, epoch, model, optimizer, loss)

            latest_ckpath = os.path.join(checkpoints_dir, "latest-model.pt")
            savecheckpoint(latest_ckpath, epoch, model, optimizer, loss)

            line = f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Time Taken: {(time.time()-stime): .01f}\n" 
            print(line)
            f.write(line)


            mae_energy, mae_force = evaluate(model, valloader, device=device)
            writer.add_scalar('MAE-Energy/val', mae_energy, epoch)
            writer.add_scalar('MAE-Force/val', mae_force, epoch)
            
            if kaggle:
                wandb.log({
                    "train_loss": loss,
                    "MAE_Energy/val": mae_energy,
                    "MAE_Force/val": mae_force,
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
                wandb.save(checkpoint_path)
                wandb.save(latest_ckpath)
            
            

    writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train NequIP model.')
    parser.add_argument('--data_dir', default="/content/drive/My Drive/MS-Physics/ML-DFT/NequIP/Data",
                       type=str, help='Base directory for data.')
    parser.add_argument('--results_dir', default="/content/drive/My Drive/MS-Physics/ML-DFT/NequIP/",
                       type=str, help='Base directory for checkpoints.')
    parser.add_argument('--mps', default=False, type=bool, help='Specify if the code is running on Mac/Cuda.')
    parser.add_argument('--finetune', default=False, type=bool, help='Fine-tune from a ' \
    'pre-trained model.')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training.')
    parser.add_argument('--checkpoint_ft', default="model_E0.pt", type=str, help='Checkpoint to ' \
    'load from when finetuning')
    parser.add_argument('--kaggle', default=False, type=bool, help='Specify if training is ' \
    'running on kaggle so as to save on wandb')
    args = parser.parse_args()


    train(args.data_dir, args.results_dir, finetune=args.finetune, batch_size=args.batch_size,
          checkpoint_ft=args.checkpoint_ft, mps=args.mps, kaggle=args.kaggle)