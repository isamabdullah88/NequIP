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
    print("--> Auto-initializing shift/scale from FULL dataset...")
    
    # Get the device of the model to ensure tensors match
    device = next(model.parameters()).device
    
    # 1. Accumulate A and y lists
    A_list = []
    y_list = []
    
    # We loop over the loader without gradients to save memory
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            # --- Build A Matrix for this batch ---
            # A[i, j] = count of atom type j in graph i
            num_species = model.output_block.shift.shape[0]
            num_graphs = batch.num_graphs
            
            # Create a local A matrix for this batch: [Batch_Size, Num_Species]
            A_batch = torch.zeros(num_graphs, num_species, device=device)
            
            # One-hot encode atom types (z)
            # shape: [Total_Atoms_In_Batch, Num_Species]
            one_hot = torch.nn.functional.one_hot(batch.z, num_species).float()
            
            # Sum atoms into their respective graphs
            # batch.batch maps each atom to its graph index (0, 0, 0, 1, 1, ...)
            A_batch.index_add_(0, batch.batch, one_hot)
            
            # --- Store ---
            A_list.append(A_batch)
            y_list.append(batch.y)
            
    # 2. Concatenate into one giant system of linear equations
    # Shape: [Total_Dataset_Size, Num_Species]
    A_full = torch.cat(A_list, dim=0)
    y_full = torch.cat(y_list, dim=0)
    
    print(f"--> Solving Linear System for {A_full.shape[0]} structures...")
    
    # 3. Solve Least Squares (Ax = y)
    # This finds the "Average Energy per Atom" for each species globally
    try:
        # lstsq returns (solution, residuals, rank, singular_values)
        solution = torch.linalg.lstsq(A_full, y_full).solution.flatten()
        
        # 4. Update the Model
        model.output_block.shift.data = solution
        print(f"--> Success! Initialized shifts: {solution}")
        
    except Exception as e:
        print(f"--> LSTSQ Failed: {e}")
        print("--> Fallback: Using Mean Energy / Mean Atoms")
        # Simple fallback: Mean Energy / Mean Number of Atoms
        mean_energy = torch.mean(y_full)
        mean_atoms = torch.mean(A_full.sum(dim=1))
        fallback_value = mean_energy / mean_atoms
        model.output_block.shift.data = torch.ones(num_species, device=device) * fallback_value



def savecheckpoint(checkpoints_dir, epoch, model, optimizer, loss):
    checkpoint_path = os.path.join(checkpoints_dir, f"model_E{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss.item()
    }, checkpoint_path)
    print(f"✅ Saved checkpoint: {checkpoint_path}")
