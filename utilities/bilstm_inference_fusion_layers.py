import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------
# INFERENCE CONFIGURATION
# -----------------------------

# Specify the folders to test and their number of conv layers.
INFERENCE_CONFIG = {
    "densenet121": {"num_conv_layers": 120, "folder": "DenseNet-121"},
    # "densenet169": {"num_conv_layers": 168, "folder": "DenseNet-169"},
    # "densenet201": {"num_conv_layers": 200, "folder": "DenseNet-201"},
    # "gpt2": {"num_conv_layers": 0, "folder": "gpt2"},
    # "3D": {"num_conv_layers": 0, "folder": "3D"},
    # "paligemma": {"num_conv_layers": 1, "folder": "paligemma"},
    # "gemma3-1b": {"num_conv_layers": 0, "folder": "gemma3-1b"},
    # "gru": {"num_conv_layers": 0, "folder": "gru"},
}

# Base paths (must match the ones used for training)
DATA_DIRS = ["src/Attack/TEX", "src/Attack/Async"]

# Path to the trained model
MODEL_PATH = "layer_count_regressor_best.pt"

# Parameters (must match the ones used for training)
SEQ_LEN = 256
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Inference Dataset
# -----------------------------
class InferenceDataset(Dataset):
    """
    Dataset for inference on a single CSV pair.
    Loads the whole file, with no train/val split.
    """
    def __init__(self, tex_csv, async_csv, seq_len=256, stride=16):
        self.seq_len = seq_len
        self.stride = stride
        self.sequences = []

        # Load the data
        df_tex = pd.read_csv(tex_csv)
        df_async = pd.read_csv(async_csv)

        self.latencies_tex = df_tex['latency_cycles'].values
        self.latencies_async = df_async['latency_cycles'].values

        # Check length
        if len(self.latencies_tex) < seq_len or len(self.latencies_async) < seq_len:
            print(f"Warning: file too short to build any sequence")
            return

        # Generate every possible sequence
        num_seqs = min(len(self.latencies_tex) - seq_len, len(self.latencies_async) - seq_len)

        for i in range(0, num_seqs, stride):
            self.sequences.append(i)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        start_idx = self.sequences[idx]

        # Extract sequences
        seq_tex = self.latencies_tex[start_idx:start_idx + self.seq_len]
        seq_async = self.latencies_async[start_idx:start_idx + self.seq_len]

        # Normalize
        seq_tex = (seq_tex - seq_tex.mean()) / (seq_tex.std() + 1e-8)
        seq_async = (seq_async - seq_async.mean()) / (seq_async.std() + 1e-8)

        return (torch.tensor(seq_tex, dtype=torch.float32),
                torch.tensor(seq_async, dtype=torch.float32))

# -----------------------------
# Model (must be identical to the training one)
# -----------------------------
class DualStreamBiLSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()

        self.lstm_tex = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, bidirectional=True)
        self.lstm_async = nn.LSTM(input_size, hidden_size, num_layers,
                                  batch_first=True, bidirectional=True)

        self.fc_fusion = nn.Linear(hidden_size * 4, 128)
        self.bn = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.fc_out = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x_tex, x_async):
        x_tex = x_tex.unsqueeze(-1)
        x_async = x_async.unsqueeze(-1)

        out_tex, _ = self.lstm_tex(x_tex)
        feat_tex = out_tex[:, -1, :]

        out_async, _ = self.lstm_async(x_async)
        feat_async = out_async[:, -1, :]

        combined = torch.cat([feat_tex, feat_async], dim=-1)

        x = torch.relu(self.fc_fusion(combined))
        x = self.bn(x)
        x = self.dropout(x)
        out = self.fc_out(x)
        out = self.relu(out)

        return out.squeeze(-1)

# -----------------------------
# Find all CSVs in a folder and build the correct pairs
# -----------------------------
def find_csv_pairs_in_folder(data_dirs, folder_name):
    """
    Finds every CSV file in a given folder and builds the correct pairs.
    Pairs files by name: async_results_* with tex_results_*.
    Returns: (list of (tex_file, async_file) pairs, list of common keys)
    """
    tex_files = []
    async_files = []

    for data_dir in data_dirs:
        if 'TEX' in data_dir:
            pattern = os.path.join(data_dir, "**", "plots", folder_name, "*.csv")
            found = glob.glob(pattern, recursive=True)
            tex_files.extend(found)
        elif 'Async' in data_dir:
            pattern = os.path.join(data_dir, "**", "plots", folder_name, "*.csv")
            found = glob.glob(pattern, recursive=True)
            async_files.extend(found)

    print(f"    TEX files found: {len(tex_files)}")
    print(f"    Async files found: {len(async_files)}")

    # Build a dict for pairing based on file name
    tex_dict = {}
    for tex_file in tex_files:
        basename = os.path.basename(tex_file)
        if basename.startswith('tex_results_'):
            # Extract the part after 'tex_results_'
            key = basename.replace('tex_results_', '')
            tex_dict[key] = tex_file

    async_dict = {}
    for async_file in async_files:
        basename = os.path.basename(async_file)
        if basename.startswith('async_results_'):
            # Extract the part after 'async_results_'
            key = basename.replace('async_results_', '')
            async_dict[key] = async_file

    # Find the common keys (files present in both TEX and Async)
    common_keys = set(tex_dict.keys()) & set(async_dict.keys())

    if not common_keys and (tex_files or async_files):
        print(f"    WARNING: no matching pairs found!")
        print(f"    TEX key examples: {list(tex_dict.keys())[:3]}")
        print(f"    Async key examples: {list(async_dict.keys())[:3]}")

    # Build the pairs
    pairs = []
    for key in sorted(common_keys):  # sorted for a consistent order
        pairs.append((tex_dict[key], async_dict[key]))

    return pairs, list(sorted(common_keys))

# -----------------------------
# Inference function
# -----------------------------
def run_inference(model, tex_csv, async_csv, model_name, true_num_conv):
    """
    Runs inference on a single CSV pair.
    """
    print(f"\n{'='*60}")
    print(f"Inference on: {model_name}")
    print(f"True number of conv layers: {true_num_conv}")
    print(f"{'='*60}")
    print(f"TEX file: {os.path.basename(tex_csv)}")
    print(f"Async file: {os.path.basename(async_csv)}")

    # Build dataset
    dataset = InferenceDataset(tex_csv, async_csv, seq_len=SEQ_LEN, stride=256)

    if len(dataset) == 0:
        print("No sequences generated, skipping this file")
        return None

    print(f"Sequences generated: {len(dataset)}")

    # DataLoader
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Inference
    model.eval()
    predictions = []

    with torch.no_grad():
        for x_tex, x_async in tqdm(loader, desc="  Inference", unit="batch"):
            x_tex = x_tex.to(DEVICE)
            x_async = x_async.to(DEVICE)

            outputs = model(x_tex, x_async)
            predictions.extend(outputs.cpu().numpy())

    predictions = np.array(predictions)

    # Statistics
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    median_pred = np.median(predictions)
    min_pred = np.min(predictions)
    max_pred = np.max(predictions)

    # Errors
    abs_error = abs(mean_pred - true_num_conv)
    relative_error = (abs_error / (true_num_conv + 1e-8)) * 100  # avoid division by zero

    print(f"\n{'-'*60}")
    print("Results:")
    print(f"{'-'*60}")
    print(f"  Mean prediction:      {mean_pred:.2f} layers")
    print(f"  Median prediction:    {median_pred:.2f} layers")
    print(f"  Standard deviation:   {std_pred:.2f}")
    print(f"  Min prediction:       {min_pred:.2f}")
    print(f"  Max prediction:       {max_pred:.2f}")
    print(f"\n  True value:           {true_num_conv} layers")
    print(f"  Absolute error:       {abs_error:.2f} layers")
    print(f"  Relative error:       {relative_error:.2f}%")

    return {
        'model': model_name,
        'true_num_conv': true_num_conv,
        'predictions': predictions,
        'mean': mean_pred,
        'median': median_pred,
        'std': std_pred,
        'min': min_pred,
        'max': max_pred,
        'abs_error': abs_error,
        'relative_error': relative_error,
        'tex_file': os.path.basename(tex_csv),
        'async_file': os.path.basename(async_csv)
    }

# -----------------------------
# Results visualization
# -----------------------------
def plot_inference_results(results):
    """
    Create visualizations of inference results.
    """
    if not results:
        print("No results to visualize")
        return

    num_results = len(results)

    # Main figure - distributions for each file
    fig = plt.figure(figsize=(20, 5 * ((num_results + 1) // 2)))

    for idx, result in enumerate(results):
        plt.subplot((num_results + 1) // 2, 2, idx + 1)

        predictions = result['predictions']
        true_value = result['true_num_conv']

        # Prediction histogram
        plt.hist(predictions, bins=50, alpha=0.7, edgecolor='black', label='Predictions')

        # Vertical lines
        plt.axvline(x=result['mean'], color='blue', linestyle='--', linewidth=2,
                   label=f"Mean: {result['mean']:.1f}")
        plt.axvline(x=result['median'], color='green', linestyle='--', linewidth=2,
                   label=f"Median: {result['median']:.1f}")
        plt.axvline(x=true_value, color='red', linestyle='-', linewidth=2,
                   label=f"True value: {true_value}")

        plt.xlabel("Predicted number of conv layers", fontsize=11)
        plt.ylabel("Frequency", fontsize=11)
        plt.title(
            f"{result['model']} - {result['tex_file']}\n"
            f"Error: {result['abs_error']:.2f} layers ({result['relative_error']:.1f}%)",
            fontsize=12, fontweight='bold'
        )
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("inference_results_distributions.png", dpi=300, bbox_inches='tight')
    print("\nPlot saved: inference_results_distributions.png")
    plt.show()

    # Group results by model
    models_data = {}
    for r in results:
        model_name = r['model']
        if model_name not in models_data:
            models_data[model_name] = {
                'predictions': [],
                'true_value': r['true_num_conv'],
                'files': []
            }
        models_data[model_name]['predictions'].extend(r['predictions'])
        models_data[model_name]['files'].append(r['tex_file'])

    # Aggregated comparison plot by model
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: predictions vs true values (aggregated)
    ax1 = axes[0]
    models = list(models_data.keys())
    true_values = [models_data[m]['true_value'] for m in models]
    mean_preds = [np.mean(models_data[m]['predictions']) for m in models]
    std_preds = [np.std(models_data[m]['predictions']) for m in models]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax1.bar(x - width/2, true_values, width, label='True Value', alpha=0.8)
    bars2 = ax1.bar(
        x + width/2, mean_preds, width, label='Mean Prediction',
        yerr=std_preds, alpha=0.8, capsize=5
    )

    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Conv Layers', fontsize=12, fontweight='bold')
    ax1.set_title(
        'Comparison: Predictions vs True Values (Aggregated)',
        fontsize=14, fontweight='bold'
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # Subplot 2: mean absolute errors per model
    ax2 = axes[1]
    abs_errors = [
        abs(np.mean(models_data[m]['predictions']) - models_data[m]['true_value'])
        for m in models
    ]
    colors = ['green' if e < 5 else 'orange' if e < 10 else 'red' for e in abs_errors]

    bars = ax2.bar(x, abs_errors, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Absolute Error (layers)', fontsize=12, fontweight='bold')
    ax2.set_title('Absolute Errors per Model', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add values above bars
    for i, (bar, error) in enumerate(zip(bars, abs_errors)):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2., height,
            f'{error:.1f}',
            ha='center', va='bottom', fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig("inference_results_comparison.png", dpi=300, bbox_inches='tight')
    print("Plot saved: inference_results_comparison.png")
    plt.show()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("="*60)
    print("LAYER COUNT REGRESSION - INFERENCE")
    print("="*60)

    # Check the configuration
    if not INFERENCE_CONFIG:
        print("\nERROR: INFERENCE_CONFIG is empty!")
        print("Edit the INFERENCE_CONFIG section in this script to specify:")
        print("  - The model to test")
        print("  - The number of convolutional layers")
        print("  - The folder containing the CSVs")
        exit(1)

    # Check that the model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Model not found at {MODEL_PATH}")
        print("Make sure you ran training before running inference!")
        exit(1)

    # Load the model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = DualStreamBiLSTMRegressor().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully")

    # Find all CSVs for each configuration entry
    print("\n" + "="*60)
    print("FINDING CSV FILES AND BUILDING PAIRS")
    print("="*60)

    all_pairs = []

    for model_name, config in INFERENCE_CONFIG.items():
        folder = config['folder']
        num_conv = config['num_conv_layers']

        print(f"\nSearching files for: {model_name}")
        print(f"  Folder: {folder}")
        print(f"  Expected conv layers: {num_conv}")

        # Use the dedicated function to build correct pairs
        pairs, common_keys = find_csv_pairs_in_folder(DATA_DIRS, folder)

        print(f"  Valid pairs found: {len(pairs)}")

        if not pairs:
            print(f"  Skipped (no matching pairs found)")
            continue

        # Show a few pairing examples
        if pairs:
            print(f"  Pairing examples:")
            for i, (tex_file, async_file) in enumerate(pairs[:3]):
                key = common_keys[i] if i < len(common_keys) else "N/A"
                print(f"    {i+1}. {key}")
                print(f"       TEX:   {os.path.basename(tex_file)}")
                print(f"       Async: {os.path.basename(async_file)}")

        # Add every pair found
        for tex_file, async_file in pairs:
            all_pairs.append({
                'model': model_name,
                'num_conv': num_conv,
                'tex_csv': tex_file,
                'async_csv': async_file
            })

        print(f"  -> {len(pairs)} pairs added")

    if not all_pairs:
        print("\nERROR: No file pairs found!")
        print("Check:")
        print("  1. The paths in DATA_DIRS")
        print("  2. The folder names in INFERENCE_CONFIG")
        print("  3. That .csv files exist in the given folders")
        print("  4. That the file names follow the pattern:")
        print("     - tex_results_*.csv")
        print("     - async_results_*.csv")
        exit(1)

    print(f"\nTotal pairs to analyze: {len(all_pairs)}")

    # Run inference on every pair
    print("\n" + "="*60)
    print("RUNNING INFERENCE")
    print("="*60)

    all_results = []

    for pair in all_pairs:
        result = run_inference(
            model=model,
            tex_csv=pair['tex_csv'],
            async_csv=pair['async_csv'],
            model_name=pair['model'],
            true_num_conv=pair['num_conv']
        )

        if result is not None:
            all_results.append(result)

    # Save results to file
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    with open("inference_results.txt", "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("LAYER COUNT REGRESSION - INFERENCE RESULTS\n")
        f.write("="*60 + "\n\n")

        f.write("Tested models:\n")
        for model_name, config in INFERENCE_CONFIG.items():
            f.write(f"  {model_name}: {config['num_conv_layers']} conv layers (folder: {config['folder']})\n")
        f.write("\n" + "="*60 + "\n\n")

        for result in all_results:
            f.write(f"\nModel: {result['model']}\n")
            f.write(f"{'-'*60}\n")
            f.write(f"  TEX file:   {result['tex_file']}\n")
            f.write(f"  Async file: {result['async_file']}\n")
            f.write(f"  Analyzed sequences: {len(result['predictions'])}\n\n")
            f.write(f"  True value:          {result['true_num_conv']} conv layers\n")
            f.write(f"  Mean prediction:     {result['mean']:.2f} layers\n")
            f.write(f"  Median prediction:   {result['median']:.2f} layers\n")
            f.write(f"  Standard deviation:  {result['std']:.2f}\n")
            f.write(f"  Range:               [{result['min']:.2f}, {result['max']:.2f}]\n\n")
            f.write(f"  Absolute error:      {result['abs_error']:.2f} layers\n")
            f.write(f"  Relative error:      {result['relative_error']:.2f}%\n")
            f.write(f"{'-'*60}\n")

        # Overall statistics
        if all_results:
            f.write(f"\n{'='*60}\n")
            f.write("OVERALL STATISTICS\n")
            f.write(f"{'='*60}\n\n")

            # Group by model
            models_stats = {}
            for result in all_results:
                model = result['model']
                if model not in models_stats:
                    models_stats[model] = {
                        'abs_errors': [],
                        'rel_errors': [],
                        'num_files': 0,
                        'true_value': result['true_num_conv']
                    }
                models_stats[model]['abs_errors'].append(result['abs_error'])
                models_stats[model]['rel_errors'].append(result['relative_error'])
                models_stats[model]['num_files'] += 1

            for model_name, stats in models_stats.items():
                mean_abs_error = np.mean(stats['abs_errors'])
                mean_rel_error = np.mean(stats['rel_errors'])

                f.write(f"{model_name}:\n")
                f.write(f"  Tested files:          {stats['num_files']}\n")
                f.write(f"  True value:            {stats['true_value']} layers\n")
                f.write(f"  Mean absolute error:   {mean_abs_error:.2f} layers\n")
                f.write(f"  Mean relative error:   {mean_rel_error:.2f}%\n\n")

    print("\nResults saved to: inference_results.txt")

    # Print on-screen summary
    print(f"\nSummary by model:")
    models_stats = {}
    for result in all_results:
        model = result['model']
        if model not in models_stats:
            models_stats[model] = {'abs_errors': [], 'num_files': 0}
        models_stats[model]['abs_errors'].append(result['abs_error'])
        models_stats[model]['num_files'] += 1

    for model_name, stats in models_stats.items():
        mean_abs_error = np.mean(stats['abs_errors'])
        print(f"  {model_name}:")
        print(f"    Tested files: {stats['num_files']}")
        print(f"    Mean MAE: {mean_abs_error:.2f} layers")

    # Create visualizations
    if all_results:
        print("\n" + "="*60)
        print("GENERATING PLOTS")
        print("="*60)
        plot_inference_results(all_results)

    print("\n" + "="*60)
    print("INFERENCE COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - inference_results.txt")
    print("  - inference_results_distributions.png")
    print("  - inference_results_comparison.png")
