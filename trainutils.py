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
    print("--> Auto-initializing shift/scale...")
    
    # Get the device where the model lives (cuda, mps, or cpu)
    model_device = next(model.parameters()).device
    num_species = model.output_block.shift.shape[0]
    
    # Define Map: Aspirin (H=1, C=6, O=8) -> (0, 1, 2)
    # Add more if you have N(7), etc.
    z_map = {1: 0, 6: 1, 7: 2, 8: 2} # Adjust '8' to '2' or '3' based on your species count!
    
    A_list = []
    y_list = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Force to CPU for safe math
            batch = batch.cpu()
            
            # --- SMART MAPPING LOGIC ---
            # Check if data is already indices (0, 1, 2) or Atomic nums (1, 6, 8)
            z_values = batch.z
            
            # If the max value in z is small (e.g., < 3), it's likely already indices.
            if z_values.max() < num_species:
                mapped_z = z_values.long() # No map needed
            else:
                # It has big numbers (1, 6, 8), so we MUST map
                try:
                    mapped_z = torch.tensor([z_map.get(int(z), -1) for z in z_values])
                except:
                    print("Error in mapping loop.")
                    return model
                
                # Check for failure (unmapped atoms)
                if (mapped_z == -1).any():
                    invalid_atom = z_values[mapped_z == -1][0].item()
                    print(f"!!! Error: Dataset contains Atom Type '{invalid_atom}' which is not in z_map.")
                    print(f"!!! Please update z_map in the code.")
                    return model

            # --- Build Matrix ---
            one_hot = torch.nn.functional.one_hot(mapped_z, num_species).float()
            A_batch = torch.zeros(batch.num_graphs, num_species)
            A_batch.index_add_(0, batch.batch, one_hot)
            
            A_list.append(A_batch)
            y_list.append(batch.y)

    # Solve on CPU
    A_full = torch.cat(A_list, dim=0)
    y_full = torch.cat(y_list, dim=0)
    
    print(f"--> Solving for {len(y_full)} structures on CPU...")
    
    try:
        # 'gels' is the most robust driver for CPU
        solution = torch.linalg.lstsq(A_full, y_full, driver="gels").solution.flatten()
        print(f"--> Success! Shifts: {solution}")
        
        # Move result to the correct device (MPS/CUDA)
        model.output_block.shift.data = solution.to(model_device)
        
    except Exception as e:
        print(f"--> Solver Failed: {e}")
        # Fallback
        mean_val = y_full.mean() / A_full.sum(1).mean()
        model.output_block.shift.data = torch.ones(num_species, device=model_device) * mean_val

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
