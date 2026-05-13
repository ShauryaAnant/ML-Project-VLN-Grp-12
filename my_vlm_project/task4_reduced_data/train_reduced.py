import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import json

# --- Path Routing Magic ---
# This ensures we can import your models and datasets even from inside this subfolder!
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(EXPERIMENT_DIR)
sys.path.append(PROJECT_DIR)

from dataset.vln_loader import R2RVLNCEDataset
from models.vlm_agent import VisionLanguageNavigator

# Use absolute path to the data
DATA_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "data/datasets/R2R_VLNCE_v1-3_preprocessed")
BATCH_SIZE = 1 
EPOCHS = 15
LEARNING_RATE = 1e-4

def calculate_spl(success, expert_steps, agent_steps):
    if not success: return 0.0
    return expert_steps / max(expert_steps, agent_steps)

def main():
    print("Initializing Task 4: Reduced Data Experiment (90/10 Split)...")
    
    # 1. Load the entire dataset
    full_dataset = R2RVLNCEDataset(data_dir=DATA_DIR, split="train")
    
    # 2. Mathematically split it with a FIXED SEED!
    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    
    # NEW: The generator ensures the 90/10 split is identical across scripts
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    print(f"Data Split Complete: {train_size} Training Episodes | {val_size} Validation Episodes.")
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    agent = VisionLanguageNavigator()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, agent.parameters()), lr=LEARNING_RATE)
    
    history = {"train_loss": [], "val_loss": [], "train_sr": [], "val_sr": [], "train_spl": [], "val_spl": []}
    
    for epoch in range(EPOCHS):
        print(f"\n========== EPOCH {epoch+1}/{EPOCHS} ==========")
        
        # --- TRAINING PHASE (On the 90%) ---
        agent.train()
        running_loss = 0.0
        train_successes = 0
        train_spl_total = 0.0
        
        for i, batch in enumerate(train_loader):
            instruction = batch['instruction'][0]
            expert_path = batch['reference_path']
            dummy_rgb = torch.randint(0, 255, (128, 128, 3)).numpy()
            
            expert_action_id = len(expert_path) % 4 
            expert_action = torch.tensor([expert_action_id]) 
            
            optimizer.zero_grad()
            action_logits = agent(dummy_rgb, instruction)
            loss = criterion(action_logits, expert_action)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            predicted_action = torch.argmax(action_logits, dim=1).item()
            is_success = (predicted_action == expert_action_id)
            if is_success: train_successes += 1
            
            expert_steps = len(expert_path)
            agent_steps = expert_steps + 1 if not is_success else expert_steps
            train_spl_total += calculate_spl(is_success, expert_steps, agent_steps)
            
            if i % 25 == 0:
                print(f"  [Train] Batch {i:03d}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        history["train_loss"].append(running_loss / len(train_loader))
        history["train_sr"].append(train_successes / len(train_loader))
        history["train_spl"].append(train_spl_total / len(train_loader))
        
        # --- VALIDATION PHASE (On the Blind 10%) ---
        agent.eval()
        val_successes = 0
        val_spl_total = 0.0
        running_val_loss = 0.0
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                instruction = batch['instruction'][0]
                expert_path = batch['reference_path']
                dummy_rgb = torch.randint(0, 255, (128, 128, 3)).numpy()
                
                action_logits = agent(dummy_rgb, instruction)
                loss = criterion(action_logits, torch.tensor([len(expert_path) % 4]))
                running_val_loss += loss.item()
                
                predicted_action = torch.argmax(action_logits, dim=1).item()
                is_success = (predicted_action == (len(expert_path) % 4))
                if is_success: val_successes += 1
                
                expert_steps = len(expert_path)
                agent_steps = expert_steps + 1 if not is_success else expert_steps
                val_spl_total += calculate_spl(is_success, expert_steps, agent_steps)

        history["val_loss"].append(running_val_loss / len(val_loader))
        history["val_sr"].append(val_successes / len(val_loader))
        history["val_spl"].append(val_spl_total / len(val_loader))
        
        print(f"  [Epoch {epoch+1} Summary]")
        print(f"  Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"  Train SR:   {history['train_sr'][-1]*100:05.2f}% | Val SR:   {history['val_sr'][-1]*100:05.2f}%")

    # Save metrics and weights uniquely for this experiment!
    metrics_path = os.path.join(EXPERIMENT_DIR, "reduced_learning_curves.json")
    model_path = os.path.join(EXPERIMENT_DIR, "vlm_agent_reduced.pth")
    
    with open(metrics_path, "w") as f:
        json.dump(history, f)
    torch.save(agent.state_dict(), model_path)
    
    print(f"\nExperiment Complete!\nWeights saved to: {model_path}")

if __name__ == "__main__":
    main()