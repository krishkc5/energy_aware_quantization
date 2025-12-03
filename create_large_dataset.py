"""
Generate a larger tokenized dataset for energy measurements.
This creates 500 or 1000 samples to minimize data reuse during experiments.
"""

import torch
from transformers import DistilBertTokenizer
from datasets import load_dataset
from pathlib import Path
import json


def prepare_tokenized_dataset(
    num_samples: int = 500,
    max_length: int = 128,
    dataset_name: str = "sst2",
    output_dir: str = r"c:\Users\taara\UPENN JR FALL\ESE 5390\energy_aware_quantization\datasets\tokenized_data_xlarge",
    seed: int = 42
):
    """
    Pre-tokenize dataset and save to disk.

    Args:
        num_samples: Number of examples to tokenize (500-1000 recommended for energy experiments)
        max_length: Maximum sequence length (128 is good for DistilBERT)
        dataset_name: Which GLUE task to use ("sst2", "mnli")
        output_dir: Directory to save tokenized data
        seed: Random seed for reproducibility
    """

    print("="*60)
    print("Pre-tokenizing Dataset for Energy Measurement")
    print("="*60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Load tokenizer
    print("\n[1/5] Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Load dataset
    print(f"[2/5] Loading {dataset_name} validation set...")
    if dataset_name == "sst2":
        dataset = load_dataset("glue", "sst2", split="validation")
        text_key = "sentence"
        label_key = "label"
        num_labels = 2
    elif dataset_name == "mnli":
        dataset = load_dataset("glue", "mnli", split="validation_matched")
        text_key = "premise"
        label_key = "label"
        num_labels = 3
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    # Check if we have enough samples
    total_available = len(dataset)
    print(f"    Total available samples: {total_available}")

    if num_samples > total_available:
        print(f"    WARNING: Requested {num_samples} but only {total_available} available")
        print(f"    Using all {total_available} samples")
        num_samples = total_available

    # Sample examples
    print(f"[3/5] Selecting {num_samples} examples (seed={seed})...")
    dataset = dataset.shuffle(seed=seed).select(range(num_samples))

    # Tokenize all examples
    print(f"[4/5] Tokenizing with max_length={max_length}...")
    texts = [example[text_key] for example in dataset]
    labels = [example[label_key] for example in dataset]

    # Tokenize in batch
    encodings = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Save tensors
    print(f"[5/5] Saving to {output_dir}...")
    torch.save(encodings['input_ids'], output_path / 'input_ids.pt')
    torch.save(encodings['attention_mask'], output_path / 'attention_mask.pt')
    torch.save(labels_tensor, output_path / 'labels.pt')

    # Save metadata
    metadata = {
        'num_samples': num_samples,
        'max_length': max_length,
        'dataset_name': dataset_name,
        'num_labels': num_labels,
        'seed': seed,
        'tokenizer': 'distilbert-base-uncased',
    }

    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("Dataset Preparation Complete!")
    print("="*60)
    print(f"Number of samples:     {num_samples}")
    print(f"Max sequence length:   {max_length}")
    print(f"Dataset:               {dataset_name}")
    print(f"Number of labels:      {num_labels}")
    print(f"\nSaved files:")
    print(f"  - input_ids.pt       {encodings['input_ids'].shape}")
    print(f"  - attention_mask.pt  {encodings['attention_mask'].shape}")
    print(f"  - labels.pt          {labels_tensor.shape}")
    print(f"  - metadata.json")
    print(f"\nOutput directory: {output_dir}")
    print("="*60)

    # Calculate file sizes
    import os
    total_size = 0
    for file in ['input_ids.pt', 'attention_mask.pt', 'labels.pt', 'metadata.json']:
        file_path = output_path / file
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        total_size += size_mb
        print(f"  {file:20s}: {size_mb:.2f} MB")
    print(f"  {'Total':20s}: {total_size:.2f} MB")

    return metadata


if __name__ == "__main__":
    print("Creating extra-large tokenized dataset for energy measurements\n")

    # Option 1: 500 samples (moderate size)
    print("\n" + "="*70)
    print("Creating 500-sample dataset (recommended for most experiments)")
    print("="*70)
    metadata_500 = prepare_tokenized_dataset(
        num_samples=500,
        max_length=128,
        dataset_name='sst2',
        output_dir=r"c:\Users\taara\UPENN JR FALL\ESE 5390\energy_aware_quantization\datasets\tokenized_data_500",
        seed=42
    )

    print("\n\n" + "="*70)
    print("Dataset created successfully!")
    print("="*70)
    print("\nUsage in notebook:")
    print(r'  dataset_path: str = r"c:\Users\taara\UPENN JR FALL\ESE 5390\energy_aware_quantization\datasets\tokenized_data_500"')
    print("\nWith 500 samples:")
    print("  - batch_size=32, num_loops=1000 = 32,000 samples → 64x reuse")
    print("  - batch_size=16, num_loops=1000 = 16,000 samples → 32x reuse")
    print("  - batch_size=32, num_loops=500  = 16,000 samples → 32x reuse")

    # Optionally create 872 samples (all validation data)
    print("\n\n" + "="*70)
    print("Do you want to create the FULL validation dataset (872 samples)?")
    print("This will use ALL available SST-2 validation data.")
    print("="*70)
    response = input("Create full dataset? (y/n): ").lower().strip()

    if response == 'y':
        metadata_full = prepare_tokenized_dataset(
            num_samples=872,  # All SST-2 validation samples
            max_length=128,
            dataset_name='sst2',
            output_dir=r"c:\Users\taara\UPENN JR FALL\ESE 5390\energy_aware_quantization\datasets\tokenized_data_full",
            seed=42
        )
        print("\n✓ Full dataset created!")

    print("\n" + "="*70)
    print("All done! Update your notebook config to use the new dataset path.")
    print("="*70)
