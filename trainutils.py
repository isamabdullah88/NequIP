import os
import torch
from model import NequIP

def loadmodel(checkpoint_path, mps):
    # Load model architecture
    model = NequIP(mps=mps)
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
    print("--> Initializing shifts (CPU Mode)...")
    
    A_list, y_list = [], []
    
    # Define Map: Aspirin has H=1, C=6, O=8 -> convert to indices 0, 1, 2
    # If you use a different molecule, update this map!
    z_map = {1: 0, 6: 1, 8: 2}

    with torch.no_grad():
        for batch in data_loader:
            # 1. Force everything to CPU (Fixes CUDA solver errors)
            batch = batch.cpu()
            
            # 2. Fix the "Index Mismatch" bug
            # Maps [1, 6, 8] to [0, 1, 2] so one_hot works correctly
            try:
                mapped_z = torch.tensor([z_map[int(z)] for z in batch.z])
            except KeyError as e:
                print(f"Error: Unknown atom type {e}. Update z_map!")
                return
            
            # 3. Build Linear System (Ax = y)
            num_species = 3  # H, C, O
            one_hot = torch.nn.functional.one_hot(mapped_z, num_species).float()
            
            # Count atoms per graph
            A_batch = torch.zeros(batch.num_graphs, num_species)
            A_batch.index_add_(0, batch.batch, one_hot)
            
            A_list.append(A_batch)
            y_list.append(batch.y)

    # 4. Solve on CPU
    A_full = torch.cat(A_list, dim=0)
    y_full = torch.cat(y_list, dim=0)
    
    print(f"--> Solving for {len(y_full)} structures...")
    
    try:
        # 'gels' driver is very stable on CPU
        solution = torch.linalg.lstsq(A_full, y_full, driver="gels").solution.flatten()
        print(f"--> Success! Shifts: {solution}")
        
        # Move result back to GPU for the model
        device = next(model.parameters()).device
        model.output_block.shift.data = solution.to(device)
        
    except Exception as e:
        print(f"--> Solver Failed: {e}")

    return model



def savecheckpoint(checkpoints_dir, epoch, model, optimizer, loss):
    checkpoint_path = os.path.join(checkpoints_dir, f"model_E{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss.item()
    }, checkpoint_path)
    print(f"✅ Saved checkpoint: {checkpoint_path}")
