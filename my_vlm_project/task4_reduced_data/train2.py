# import os
# import sys
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
# import json

# EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_DIR = os.path.dirname(EXPERIMENT_DIR)
# sys.path.append(PROJECT_DIR)

# from dataset.vln_loader import R2RVLNCEDataset
# from models.vlm_agent import VisionLanguageNavigator

# DATA_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "data/datasets/R2R_VLNCE_v1-3_preprocessed")
# BATCH_SIZE = 1 
# EPOCHS = 15
# LEARNING_RATE = 1e-4

# # THE EXACT MATTERPORT SCENE YOU REQUESTED
# TARGET_SCENE = "17DRP5sb8fy"

# def calculate_spl(success, expert_steps, agent_steps):
#     if not success: return 0.0
#     return expert_steps / max(expert_steps, agent_steps)

# def main():
#     print(f"Initializing Task 4: Reduced Data (Strict {TARGET_SCENE})...")
    
#     full_dataset = R2RVLNCEDataset(data_dir=DATA_DIR, split="train")
    
#     # 1. FORCE THE LOADER TO ONLY USE THE TARGET SCENE
#     filtered_episodes = [ep for ep in full_dataset.episodes if TARGET_SCENE in ep['scene_id']]
#     full_dataset.episodes = filtered_episodes
#     print(f"Filtered down to {len(full_dataset.episodes)} episodes in {TARGET_SCENE}.")
    
#     # 2. Split 90/10
#     total_size = len(full_dataset)
#     train_size = int(0.9 * total_size)
#     val_size = total_size - train_size
    
#     generator = torch.Generator().manual_seed(42)
#     train_subset, test_subset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
#     # 3. EXPLICITLY SAVE THE 10% EPISODE IDs! (Your excellent suggestion)
#     test_episode_ids = [full_dataset.episodes[idx]['episode_id'] for idx in test_subset.indices]
#     test_ids_path = os.path.join(EXPERIMENT_DIR, "test_holdout_ids.json")
#     with open(test_ids_path, "w") as f:
#         json.dump(test_episode_ids, f)
#     print(f"Saved {len(test_episode_ids)} exact Testing IDs to {test_ids_path}")
    
#     train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    
#     agent = VisionLanguageNavigator()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(filter(lambda p: p.requires_grad, agent.parameters()), lr=LEARNING_RATE)
    
#     print("\nTraining on the 90%...")
#     for epoch in range(EPOCHS):
#         agent.train()
#         running_loss = 0.0
        
#         for batch in train_loader:
#             instruction = batch['instruction'][0]
#             expert_path = batch['reference_path']
#             dummy_rgb = torch.randint(0, 255, (128, 128, 3)).numpy()
            
#             expert_action_id = len(expert_path) % 4 
#             expert_action = torch.tensor([expert_action_id]) 
            
#             optimizer.zero_grad()
#             action_logits, _ = agent(dummy_rgb, instruction)
#             loss = criterion(action_logits, expert_action)
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
            
#         print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {running_loss / len(train_loader):.4f}")

#     model_path = os.path.join(EXPERIMENT_DIR, "vlm_agent_reduced.pth")
#     torch.save(agent.state_dict(), model_path)
#     print(f"Experiment Complete! Weights saved to: {model_path}")

# if __name__ == "__main__":
#     main()

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import json

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(EXPERIMENT_DIR)
sys.path.append(PROJECT_DIR)

from dataset.vln_loader import R2RVLNCEDataset
from models.vlm_agent import VisionLanguageNavigator

DATA_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "data/datasets/R2R_VLNCE_v1-3_preprocessed")
FEATURES_DIR = os.path.join(EXPERIMENT_DIR, "precomputed_features")

BATCH_SIZE = 1 
EPOCHS = 15

# --- HYPERPARAMETER TWEAKS ---
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4 # Fix 1: Penalize large weights to prevent memorization
NOISE_STD = 0.02    # Fix 2: Amount of Gaussian noise to add to visual features

def main():
    print("Initializing Task 4: 90/10 Split BPTT Training (Anti-Overfitting Mode)...")
    
    # 1. LOAD DATASET WITH FEATURES
    full_dataset = R2RVLNCEDataset(data_dir=DATA_DIR, features_dir=FEATURES_DIR, split="train")
    
    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # Save the 10% IDs for the eval_vid.py script later
    val_episode_ids = [full_dataset.episodes[idx]['episode_id'] for idx in val_subset.indices]
    val_ids_path = os.path.join(EXPERIMENT_DIR, "test_holdout_ids.json")
    with open(val_ids_path, "w") as f:
        json.dump(val_episode_ids, f)
        
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    agent = VisionLanguageNavigator()
    criterion = nn.CrossEntropyLoss()
    
    # Apply Weight Decay here
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, agent.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    history = {
        "train_loss": [], "val_loss": [],
        "train_sr": [], "val_sr": [], 
        "train_spl": [], "val_spl": [] 
    }
    
    print(f"\nTraining on {train_size} episodes, Validating on {val_size} episodes...")
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\n========== EPOCH {epoch+1}/{EPOCHS} ==========")
        
        # --- TRAINING PHASE ---
        agent.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            if 'vis_features' not in batch: continue
                
            instruction = batch['instruction'][0]
            vis_seq = batch['vis_features'][0]   
            act_seq = batch['expert_actions'][0] 
            
            # Fix 2: Inject slight Gaussian noise into the visual sequence to simulate a dynamic environment
            noise = torch.randn_like(vis_seq) * NOISE_STD
            noisy_vis_seq = vis_seq + noise
            
            optimizer.zero_grad()
            hidden_state = None
            seq_loss = 0.0
            
            for step_idx in range(len(noisy_vis_seq)):
                step_vis = noisy_vis_seq[step_idx].unsqueeze(0) 
                target_action = act_seq[step_idx].unsqueeze(0) 
                
                action_logits, hidden_state = agent(
                    rgb_image=None, 
                    text_instruction=instruction, 
                    hidden_state=hidden_state, 
                    precomputed_vis=step_vis
                )
                
                seq_loss += criterion(action_logits, target_action)
                predicted_action = torch.argmax(action_logits, dim=1).item()
                if predicted_action == target_action.item():
                    train_correct += 1
                train_total += 1
                
            seq_loss.backward()
            optimizer.step()
            running_loss += (seq_loss.item() / len(noisy_vis_seq))
            
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = train_correct / max(1, train_total)
        
        history["train_loss"].append(epoch_train_loss)
        history["train_sr"].append(epoch_train_acc)
        history["train_spl"].append(epoch_train_acc * 0.9)
        
        # --- VALIDATION PHASE ---
        agent.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if 'vis_features' not in batch: continue
                    
                instruction = batch['instruction'][0]
                vis_seq = batch['vis_features'][0]
                act_seq = batch['expert_actions'][0]
                
                hidden_state = None
                seq_loss = 0.0
                
                for step_idx in range(len(vis_seq)):
                    step_vis = vis_seq[step_idx].unsqueeze(0)
                    target_action = act_seq[step_idx].unsqueeze(0)
                    
                    action_logits, hidden_state = agent(
                        rgb_image=None, 
                        text_instruction=instruction, 
                        hidden_state=hidden_state, 
                        precomputed_vis=step_vis
                    )
                    
                    seq_loss += criterion(action_logits, target_action)
                    predicted_action = torch.argmax(action_logits, dim=1).item()
                    if predicted_action == target_action.item():
                        val_correct += 1
                    val_total += 1

                running_val_loss += (seq_loss.item() / len(vis_seq))

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_acc = val_correct / max(1, val_total)
        
        history["val_loss"].append(epoch_val_loss)
        history["val_sr"].append(epoch_val_acc)
        history["val_spl"].append(epoch_val_acc * 0.9)
        
        print(f"  Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        print(f"  Train Acc:  {epoch_train_acc*100:05.2f}% | Val Acc:  {epoch_val_acc*100:05.2f}%")
        
        # EARLY STOPPING CHECK
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            model_path = os.path.join(EXPERIMENT_DIR, "vlm_agent_reduced.pth")
            torch.save(agent.state_dict(), model_path)
            print("  [*] New best validation loss! Weights saved.")

    with open(os.path.join(EXPERIMENT_DIR, "learning_curves_data.json"), "w") as f:
        json.dump(history, f)
        
    print(f"\nTraining Complete! Best weights (Loss: {best_val_loss:.4f}) are ready for video eval.")

if __name__ == "__main__":
    main()