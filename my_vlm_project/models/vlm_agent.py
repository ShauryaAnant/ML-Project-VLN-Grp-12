# import torch
# import torch.nn as nn
# from transformers import CLIPProcessor, CLIPModel

# class VisionLanguageNavigator(nn.Module):
#     def __init__(self):
#         super(VisionLanguageNavigator, self).__init__()
        
#         # Load the Pretrained CLIP Model & Processor from HuggingFace
#         # We use 'clip-vit-base-patch32' as it is relatively lightweight for CPUs
#         self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#         self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
#         # FREEZE the CLIP weights! (Crucial for CPU training and Ablation studies later)
#         for param in self.clip_model.parameters():
#             param.requires_grad = False

#         # Define dimensions (CLIP ViT-B/32 outputs 512-dimensional vectors)
#         self.hidden_dim = 512

#         # -------------------------------------------------------------------
#         # TASK 2: Implement multimodal fusion module (Placeholder for now)
#         # -------------------------------------------------------------------
#         # This takes the 512 visual vector and 512 text vector and combines them
#         self.fusion_module = nn.Sequential(
#             nn.Linear(self.hidden_dim * 2, self.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.hidden_dim, self.hidden_dim)
#         )

#         # -------------------------------------------------------------------
#         # TASK 2: Implement policy head
#         # -------------------------------------------------------------------
#         # Action space: 0: Stop, 1: Move Forward, 2: Turn Left, 3: Turn Right
#         self.policy_head = nn.Sequential(
#             nn.Linear(self.hidden_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 4) # 4 possible actions
#         )

#     def forward(self, rgb_image, text_instruction):
#         """
#         rgb_image: Raw image array from Habitat sensor (H, W, C)
#         text_instruction: String, e.g., "Walk down the hall"
#         """
        
#         # 1. VISUAL ENCODER (Process image through CLIP)
#         # We use the processor to format the Habitat image exactly how CLIP expects it
#         inputs = self.processor(images=rgb_image, return_tensors="pt")
#         visual_features = self.clip_model.get_image_features(**inputs) 
        
#         # 2. TEXT ENCODER (Process string through CLIP)
#         text_inputs = self.processor(text=[text_instruction], return_tensors="pt", padding=True)
#         text_features = self.clip_model.get_text_features(**text_inputs)

#         # 3. MULTIMODAL FUSION
#         # Simplest fusion: Concatenate the two 512-d vectors into one 1024-d vector
#         fused_features = torch.cat((visual_features, text_features), dim=1)
#         fused_output = self.fusion_module(fused_features)

#         # 4. POLICY HEAD
#         # Get raw action probabilities (logits)
#         action_logits = self.policy_head(fused_output)
        
#         return action_logits

# # --- Test if it works! ---
# if __name__ == "__main__":
#     # Create the model
#     agent = VisionLanguageNavigator()
    
#     # Fake data to represent a Habitat step
#     fake_image = torch.randint(0, 255, (128, 128, 3)).numpy() # Fake Habitat RGB sensor
#     fake_instruction = "Walk past the kitchen and stop at the stairs."
    
#     # Run the forward pass
#     output_actions = agent(fake_image, fake_instruction)
#     print("Action Logits (Stop, Forward, Left, Right):", output_actions)



import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel

class VisionLanguageNavigator(nn.Module):
    def __init__(self):
        super(VisionLanguageNavigator, self).__init__()
        
        # Load the Pretrained CLIP Model & Processor from HuggingFace
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # FREEZE the CLIP weights
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.hidden_dim = 512

        # -------------------------------------------------------------------
        # UPGRADE 1: Minor Improvements to Fusion Module
        # -------------------------------------------------------------------
        self.fusion_module = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3), # IMPROVEMENT: Prevents overfitting on your small 90% dataset
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()        # IMPROVEMENT: Added non-linearity before entering the GRU
        )

        # -------------------------------------------------------------------
        # UPGRADE 2: The GRU Memory Module
        # -------------------------------------------------------------------
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # -------------------------------------------------------------------
        # TASK 2: Implement policy head
        # -------------------------------------------------------------------
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4) 
        )

    def forward(self, rgb_image, text_instruction, hidden_state=None, precomputed_vis=None):
        """
        Added 'precomputed_vis' to bypass the CLIP processor during training.
        """
        
        # 1. VISUAL ENCODER (Bypass if we have precomputed offline features)
        if precomputed_vis is not None:
            visual_features = precomputed_vis
        else:
            inputs = self.processor(images=rgb_image, return_tensors="pt")
            visual_features = self.clip_model.get_image_features(**inputs) 
        
        # 2. TEXT ENCODER
        text_inputs = self.processor(text=[text_instruction], return_tensors="pt", padding=True)
        text_features = self.clip_model.get_text_features(**text_inputs)

        # 3. MULTIMODAL FUSION
        fused_features = torch.cat((visual_features, text_features), dim=1)
        fused_output = self.fusion_module(fused_features) # Output shape: (1, 512)

        # 4. GRU MEMORY
        gru_input = fused_output.unsqueeze(1) # Reshape to (1, 1, 512)
        
        if hidden_state is None:
            gru_out, new_hidden_state = self.gru(gru_input)
        else:
            gru_out, new_hidden_state = self.gru(gru_input, hidden_state)
            
        gru_out = gru_out.squeeze(1) # Reshape back to (1, 512)

        # 5. POLICY HEAD
        action_logits = self.policy_head(gru_out)
        
        return action_logits, new_hidden_state

# --- Test if it works! ---
if __name__ == "__main__":
    agent = VisionLanguageNavigator()
    fake_image = torch.randint(0, 255, (128, 128, 3)).numpy() 
    fake_instruction = "Walk past the kitchen and stop at the stairs."
    
    print("Testing Step 1 (No Memory)...")
    output_actions, h_state = agent(fake_image, fake_instruction)
    print("Action Logits:", output_actions)
    
    print("\nTesting Step 2 (Passing Memory Forward)...")
    output_actions_2, h_state_2 = agent(fake_image, fake_instruction, hidden_state=h_state)
    print("Action Logits:", output_actions_2)
    print("\nSuccess! The GRU memory is fully integrated.")


