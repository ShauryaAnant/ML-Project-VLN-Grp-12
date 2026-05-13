import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertModel, DistilBertTokenizer

class VisionLanguageNavigator(nn.Module):
    def __init__(self, num_actions=4):
        super(VisionLanguageNavigator, self).__init__()
        
        # ---------------------------------------------------------
        # 1. PRETRAINED VISUAL ENCODER (ResNet18)
        # ---------------------------------------------------------
        # We use ResNet18 because it's fast and lightweight for an RTX 3050.
        # We strip off the final classification layer because we just want the visual features.
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.visual_feature_dim = 512 # ResNet18 outputs a 512-dimensional vector
        
        # ---------------------------------------------------------
        # 2. PRETRAINED TEXT ENCODER (DistilBERT)
        # ---------------------------------------------------------
        # DistilBERT is a smaller, faster version of BERT that understands English instructions perfectly.
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # We freeze the text encoder so it doesn't destroy your RAM during training
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        self.text_feature_dim = 768 # DistilBERT outputs a 768-dimensional vector
        
        # ---------------------------------------------------------
        # 3. MULTIMODAL FUSION MODULE
        # ---------------------------------------------------------
        # This combines the 512 visual features and the 768 text features.
        combined_dim = self.visual_feature_dim + self.text_feature_dim
        
        self.fusion_module = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # ---------------------------------------------------------
        # 4. POLICY HEAD
        # ---------------------------------------------------------
        # Takes the fused 256-dim "understanding" and maps it to the 4 possible actions:
        # 0: Stop, 1: Move Forward, 2: Turn Left, 3: Turn Right
        self.policy_head = nn.Linear(256, num_actions)

    def forward(self, images, instruction_texts):
        """
        images: Batch of RGB images from Habitat [Batch_Size, 3, Height, Width]
        instruction_texts: List of string instructions e.g., ["Go to the kitchen"]
        """
        
        # --- Process Vision ---
        # Output shape: [Batch, 512, 1, 1] -> flattened to [Batch, 512]
        v_features = self.visual_encoder(images).squeeze()
        if len(v_features.shape) == 1:
            v_features = v_features.unsqueeze(0) # Handle batch size of 1
            
        # --- Process Text ---
        # Tokenize the raw text strings into numbers DistilBERT can read
        encoded_text = self.tokenizer(
            instruction_texts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(images.device)
        
        # Extract the [CLS] token representation (the summary of the sentence)
        t_outputs = self.text_encoder(**encoded_text)
        t_features = t_outputs.last_hidden_state[:, 0, :] # Shape: [Batch, 768]
        
        # --- Multimodal Fusion ---
        # Glue the image vector and text vector together side-by-side
        fused_features = torch.cat((v_features, t_features), dim=1) # Shape: [Batch, 1280]
        fused_representation = self.fusion_module(fused_features)   # Shape: [Batch, 256]
        
        # --- Policy Decision ---
        action_logits = self.policy_head(fused_representation)      # Shape: [Batch, 4]
        
        return action_logits

# --- Quick Sanity Check ---
if __name__ == "__main__":
    # Create the model and move it to your NVIDIA GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionLanguageNavigator().to(device)
    
    # Create a fake image batch (1 image, 3 color channels, 224x224 pixels)
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    
    # Create a fake instruction
    dummy_instruction = ["Walk past the brown sofa and stop at the stairs"]
    
    # Run a forward pass
    output = model(dummy_image, dummy_instruction)
    
    print("\n✅ Model Architecture Built Successfully!")
    print(f"Using Device: {device}")
    print(f"Raw Action Logits (Stop, Forward, Left, Right): {output.detach().cpu().numpy()}")
