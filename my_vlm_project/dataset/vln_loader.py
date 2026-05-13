import os
import json
import gzip
import torch
from torch.utils.data import Dataset

class R2RVLNCEDataset(Dataset):
    def __init__(self, data_dir, features_dir=None, split="train"):
        super().__init__()
        self.data_dir = data_dir
        self.features_dir = features_dir # NEW: Path to our extracted .pt files
        self.split = split
        self.file_path = os.path.join(data_dir, f"{split}/{split}.json.gz")
        self.allowed_scenes = ['17DRP5sb8fy']
        self.episodes = self._load_and_filter_data()
        print(f"Loaded {len(self.episodes)} episodes.")

    def _load_and_filter_data(self):
        filtered_episodes = []
        try:
            with gzip.open(self.file_path, "rt") as f:
                data = json.load(f)
                
            for episode in data['episodes']:
                scene_id = episode['scene_id'].split('/')[-1].replace('.glb', '')
                if scene_id in self.allowed_scenes:
                    # NEW: Only load episodes if we successfully extracted their features
                    if self.features_dir:
                        pt_path = os.path.join(self.features_dir, f"{episode['episode_id']}.pt")
                        if os.path.exists(pt_path):
                            filtered_episodes.append(episode)
                    else:
                        filtered_episodes.append(episode)
        except FileNotFoundError:
            print("ERROR: Could not find dataset.")
        return filtered_episodes

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        episode = self.episodes[idx]
        
        result = {
            "episode_id": episode['episode_id'],
            "scene_id": episode['scene_id'],
            "instruction": episode['instruction']['instruction_text'],
            "reference_path": episode['reference_path'],
            "start_position": episode['start_position'],
            "start_rotation": episode['start_rotation']
        }
        
        # NEW: Inject the precomputed timelines!
        if self.features_dir:
            pt_path = os.path.join(self.features_dir, f"{episode['episode_id']}.pt")
            data = torch.load(pt_path, weights_only=True)
            result["vis_features"] = data["features"]     # Shape: (Steps, 512)
            result["expert_actions"] = data["actions"]    # Shape: (Steps,)
            
        return result