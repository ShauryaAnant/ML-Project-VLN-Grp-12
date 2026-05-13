# Vision-Language Reasoning for Visual Navigation Using Habitat Simulator


## Project Overview
This repository contains the implementation of a learning-based navigation system that integrates visual perception and textual reasoning to predict navigation decisions in simulation. The agent is designed to follow high-level natural-language instructions (e.g., "Go to the reception desk") and navigate complex 3D indoor environments.

The project is built on top of the **Habitat Simulator** using the **Matterport3D** dataset and focuses on the machine-learning implementation of a Vision-Language Navigation (VLN) agent using imitation learning.

## Model Architecture
Unlike traditional navigation systems relying on metric maps, this project utilizes a multimodal neural network architecture:
* **Visual Encoder:** Pretrained Vision Transformer (ViT) via the CLIP model to extract robust visual features from RGB observations.
* **Text Encoder:** Pretrained language model to process and encode natural-language navigation instructions.
* **Multimodal Fusion:** A fusion module that integrates visual and textual embeddings to ground the language instructions into the visual context.
* **Policy Head:** A discrete action predictor (move forward, turn left, turn right, stop) based on the fused representations.

## Repository Structure
The core implementation is housed in the `my_vlm_project/` directory:

```text
my_vlm_project/
├── models/
│   └── vlm_agent.py             # Core architecture (ViT/CLIP, Text Encoder, Fusion, Policy Head)
├── dataset/
│   └── vln_loader.py            # Data loading utilities for Matterport3D
├── task4_reduced_data/          # Scripts and utilities for reduced dataset ablation studies
├── extract_features.py          # Script for offline visual/text feature extraction
├── train.py                     # Main training pipeline (Imitation Learning)
├── train_reduced.py             # Training script for reduced dataset ablation study
├── eval.py / eval_vid.py        # Evaluation scripts (calculates SR and SPL)
├── plot_curves.py               # Utility to generate training visualizations
├── task4_unseen_vid.py          # Generalization testing on unseen environments
└── task3_learning_curves_3panel.png # Baseline performance visualization
```

*Note: The 15GB Matterport3D dataset, large precomputed features , paths are excluded from this repository due to size constraints.*

## Installation & Environment Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/ShauryaAnant/ML-Project-VLN-Grp-12](https://github.com/ShauryaAnant/ML-Project-VLN-Grp-12)
cd ML-Project-VLN-Grp-12
```

**2. Install Habitat Simulator:**
Ensure you are using Ubuntu/WSL. Follow the official [Habitat-Lab installation instructions](https://github.com/facebookresearch/habitat-lab) to configure the environment.

**3. Dataset Preparation:**
Place the downloaded Matterport3D scene datasets into the `data/scene_datasets/` directory at the root of the `habitat-lab` folder.

<!-- **4. Model Checkpoints:**
Download the trained weights (e.g., `vlm_agent_best.pth`) from [INSERT YOUR DRIVE LINK HERE] and place them in the `my_vlm_project/models/` directory. -->

## Usage & Execution

**1. Feature Extraction (Optional):**
```bash
python my_vlm_project/extract_features.py
```

**2. Training the Baseline Agent:**
```bash
python my_vlm_project/train.py
```

**3. Evaluating the Model:**
To calculate Success Rate (SR) and Success weighted by Path Length (SPL) on the validation set:
```bash
python my_vlm_project/eval.py
```

**4. Testing Generalization (Unseen Environments):**
```bash
python my_vlm_project/task4_unseen_vid.py
```

**5. Plotting Learning Curves:**
```bash
python my_vlm_project/plot_curves.py
```

## Evaluation & Results
The agent is evaluated based on its **Success Rate (SR)** and **Success weighted by Path Length (SPL)**. The repository includes scripts to validate performance on both seen environments and unseen environments to test generalization capabilities. 

Additional ablation studies include performance analysis with reduced training data and controlled architectural extensions. Please refer to the `task3_learning_curves_3panel.png` for the baseline training trajectories and the accompanying final PDF report for an in-depth analytical discussion.

## Team Information
* **Shaurya Anant**: 2023CSB1313
* **Kanav Puri**: 2023MCB1298
* **Vaibhav Pawar**: 2023MCB1317
