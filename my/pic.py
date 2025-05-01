import matplotlib.pyplot as plt
import numpy as np
import re


# Function to extract data from the log text
def extract_data(log_text, model_name):
    pattern = r"Epoch: (\d+) loss_train: ([\d\.]+) acc_train: ([\d\.]+) loss_val: ([\d\.]+) acc_val: ([\d\.]+) loss_test: ([\d\.]+) acc_test: ([\d\.]+)"
    matches = re.findall(pattern, log_text)

    data = {
        'epoch': [],
        'loss_train': [],
        'acc_train': [],
        'loss_val': [],
        'acc_val': [],
        'loss_test': [],
        'acc_test': []
    }

    for match in matches:
        data['epoch'].append(int(match[0]))
        data['loss_train'].append(float(match[1]))
        data['acc_train'].append(float(match[2]))
        data['loss_val'].append(float(match[3]))
        data['acc_val'].append(float(match[4]))
        data['loss_test'].append(float(match[5]))
        data['acc_test'].append(float(match[6]))

    return data


# Read the log file
with open("C:\\Users\86734\Desktop\\result\\ans.txt", 'r') as file:
    log_text = file.read()

# Split the log text into sections for each model
model_sections = {}
if "base model:" in log_text:
    base_model_text = log_text.split("base model:")[1].split("only-rope model:")[0]
    model_sections["Base Model"] = base_model_text

if "only-rope model:" in log_text:
    rope_model_text = log_text.split("only-rope model:")[1].split("base+rope model:")[0]
    model_sections["Only-RoPE Model"] = rope_model_text

if "base+rope model:" in log_text:
    base_rope_text = log_text.split("base+rope model:")[1].split("local+global rope:")[0]
    model_sections["Base+RoPE Model"] = base_rope_text

if "local+global rope:" in log_text:
    local_global_text = log_text.split("local+global rope:")[1]
    model_sections["Local+Global RoPE"] = local_global_text

# Extract data for each model
models_data = {}
for model_name, section_text in model_sections.items():
    models_data[model_name] = extract_data(section_text, model_name)

# Set a professional style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Model Training Performance Comparison", fontsize=20, y=0.98)

# Plot 1: Test Accuracy
ax = axes[0, 0]
for i, (model_name, data) in enumerate(models_data.items()):
    ax.plot(data['epoch'], data['acc_test'],
            label=model_name, color=colors[i], marker=markers[i],
            markevery=len(data['epoch']) // 10, markersize=8, linewidth=2)
ax.set_title('Test Accuracy', fontsize=16)
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)
ax.set_ylim(0.3, 0.9)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=12, loc='lower right')

# Plot 2: Validation Accuracy
ax = axes[0, 1]
for i, (model_name, data) in enumerate(models_data.items()):
    ax.plot(data['epoch'], data['acc_val'],
            label=model_name, color=colors[i], marker=markers[i],
            markevery=len(data['epoch']) // 10, markersize=8, linewidth=2)
ax.set_title('Validation Accuracy', fontsize=16)
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)
ax.set_ylim(0.3, 0.9)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=12, loc='lower right')

# Plot 3: Test Loss
ax = axes[1, 0]
for i, (model_name, data) in enumerate(models_data.items()):
    ax.plot(data['epoch'], data['loss_test'],
            label=model_name, color=colors[i], marker=markers[i],
            markevery=len(data['epoch']) // 10, markersize=8, linewidth=2)
ax.set_title('Test Loss', fontsize=16)
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Loss', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=12, loc='upper right')

# Plot 4: Training Loss
ax = axes[1, 1]
for i, (model_name, data) in enumerate(models_data.items()):
    ax.plot(data['epoch'], data['loss_train'],
            label=model_name, color=colors[i], marker=markers[i],
            markevery=len(data['epoch']) // 10, markersize=8, linewidth=2)
ax.set_title('Training Loss', fontsize=16)
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Loss', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_yscale('log')  # Using log scale for better visualization of training loss
ax.legend(fontsize=12, loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('training_metrics_comparison.png', dpi=300, bbox_inches='tight')

# Create a focused zoom on test accuracy showing the peak performance
plt.figure(figsize=(12, 8))
for i, (model_name, data) in enumerate(models_data.items()):
    # Find the epoch with maximum test accuracy
    max_acc_idx = np.argmax(data['acc_test'])
    max_acc = data['acc_test'][max_acc_idx]
    max_epoch = data['epoch'][max_acc_idx]

    plt.plot(data['epoch'], data['acc_test'],
             label=f"{model_name} (Max: {max_acc:.4f} at epoch {max_epoch})",
             color=colors[i], linewidth=2.5)
    plt.scatter(max_epoch, max_acc, color=colors[i], s=150,
                edgecolor='black', zorder=10, marker=markers[i])

plt.title('Test Accuracy Comparison with Peak Performance', fontsize=18)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Test Accuracy', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12, loc='lower right')
plt.ylim(0.75, 0.85)  # Focus on the relevant accuracy range
plt.tight_layout()
plt.savefig('peak_accuracy_comparison.png', dpi=300, bbox_inches='tight')

# Create early convergence plot (first 100 epochs)
plt.figure(figsize=(12, 8))
for i, (model_name, data) in enumerate(models_data.items()):
    # Only use data for first 100 epochs
    cutoff = next(idx for idx, e in enumerate(data['epoch']) if e > 100) if any(
        e > 100 for e in data['epoch']) else len(data['epoch'])

    plt.plot(data['epoch'][:cutoff], data['acc_test'][:cutoff],
             label=model_name, color=colors[i], marker=markers[i],
             markevery=5, markersize=8, linewidth=2)

plt.title('Early Convergence: Test Accuracy (First 100 Epochs)', fontsize=18)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Test Accuracy', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig('early_convergence.png', dpi=300, bbox_inches='tight')

# Create a figure showing overfitting tendencies
plt.figure(figsize=(12, 8))
for i, (model_name, data) in enumerate(models_data.items()):
    # Calculate the gap between training and test accuracy
    acc_gap = [train - test for train, test in zip(data['acc_train'], data['acc_test'])]

    plt.plot(data['epoch'], acc_gap,
             label=model_name, color=colors[i], linewidth=2.5)

plt.title('Overfitting Analysis: Training-Test Accuracy Gap', fontsize=18)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Training Accuracy - Test Accuracy', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12, loc='upper left')
plt.tight_layout()
plt.savefig('overfitting_analysis.png', dpi=300, bbox_inches='tight')

print("Visualization complete! All plots have been saved.")