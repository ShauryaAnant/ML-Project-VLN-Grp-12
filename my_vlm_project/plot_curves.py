import json
import matplotlib.pyplot as plt

# Load the data
with open("learning_curves_data.json", "r") as f:
    history = json.load(f)

epochs = range(1, len(history["train_loss"]) + 1)

# Create a 1x3 grid
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# --- Plot 1: Loss ---
ax1.plot(epochs, history["train_loss"], 'b.-', label="Train")
ax1.plot(epochs, history["val_loss"], 'r.-', label="Val")
ax1.set_title("Loss")
ax1.set_xlabel("Epochs")
ax1.legend()
ax1.grid(True)

# --- Plot 2: Success Rate (Accuracy) ---
ax2.plot(epochs, history["train_sr"], 'b.-', label="Train")
ax2.plot(epochs, history["val_sr"], 'r.-', label="Val")
ax2.set_title("Success Rate")
ax2.set_xlabel("Epochs")
ax2.legend()
ax2.grid(True)

# --- Plot 3: SPL ---
ax3.plot(epochs, history["train_spl"], 'b.-', label="Train")
ax3.plot(epochs, history["val_spl"], 'r.-', label="Val")
ax3.set_title("SPL")
ax3.set_xlabel("Epochs")
ax3.legend()
ax3.grid(True)

# Save and show
plt.tight_layout()
plt.savefig("task3_learning_curves_3panel.png", dpi=300)
print("Saved 3-panel learning curves to task3_learning_curves_3panel.png!")
plt.show()