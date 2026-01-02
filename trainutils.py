import os
import torch
from model import NequIP

def loadmodel(checkpoint_path):
    # Load model architecture
    model = NequIP()
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model



# def savefig(predictions, targets, epoch):
#     predictions = torch.concatenate(predictions).cpu().detach()
#     targets = torch.concatenate(targets).cpu().detach()

#     predictions = predictions.view(-1)
#     targets = targets.view(-1)
#     import matplotlib.pyplot as plt
#     plt.scatter(targets, predictions, s=10, alpha=0.7)
#     plt.plot([targets.min(), targets.max()],
#             [targets.min(), targets.max()],
#             'r--', label='Perfect prediction')

#     plt.xlabel("True Energy (eV)")
#     plt.ylabel("Predicted Energy (eV)")
#     plt.title("Predicted vs True Total Energy")
#     plt.legend()
#     # plt.show()
#     plt.savefig(f"Figures/pred_vs_true_epoch-{epoch}.png")
#     plt.close()

def initialize_shift_scale(model, data_loader):
    print("Auto-initializing shift/scale from first batch...")
    
    # 1. Grab just ONE batch of data
    batch = next(iter(data_loader))
    # batch = batch.to(next(model.parameters()).device)
    
    # 2. Count atoms in this batch
    # (Same logic: A * shift = y)
    num_species = model.output_block.shift.shape[0]
    A = torch.zeros(batch.num_graphs, num_species)
    one_hot = torch.nn.functional.one_hot(batch.z, num_species).float()
    A.index_add_(0, batch.batch, one_hot)
    
    y = batch.y
    
    # 3. Quick Least Squares Guess
    # This solves: Energy_Total = N_Carbon * E_Carbon + N_Hydrogen * E_Hydrogen ...
    try:
        guess_shifts = torch.linalg.lstsq(A, y).solution.flatten()
    except:
        # Fallback for very small batches
        guess_shifts = torch.ones(num_species) * torch.mean(y) / torch.mean(A.sum(1))

    # 4. Force the model to start here
    # We update the .data directly so the optimizer sees this as the starting point
    model.output_block.shift.data = guess_shifts
    
    # Optional: Set scale to roughly the variance (usually 1.0 is fine to start)
    # But shift is the critical one for the 10^11 problem.
    print(f"Initialized Shifts to: {guess_shifts}")



def savecheckpoint(checkpoints_dir, epoch, model, optimizer, loss):
    checkpoint_path = os.path.join(checkpoints_dir, f"model_E{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss.item()
    }, checkpoint_path)
    print(f"✅ Saved checkpoint: {checkpoint_path}")
