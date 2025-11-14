
import datasets
import pandas as pd
import json
import os
from pathlib import Path

# Load the dataset
ds = datasets.load_dataset("nbalepur/mcqa-bench-base")

# Get unique dataset values
unique_datasets = set()
for split in ds.keys():
    unique_datasets.update(ds[split]['dataset'])

print(f"Found unique datasets: {sorted(unique_datasets)}")

# Process each unique dataset
for dataset_name in sorted(unique_datasets):
    print(f"Processing dataset: {dataset_name}")
    
    # Create directory structure
    dataset_dir = Path(f"local_datasets/{dataset_name}")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter data for each split
    for split_name in ds.keys():
        print(f"  Processing split: {split_name}")
        
        # Filter the split data for this specific dataset
        filtered_data = ds[split_name].filter(lambda x: x['dataset'] == dataset_name)
        
        if len(filtered_data) == 0:
            print(f"    No data found for {dataset_name} in {split_name} split")
            continue
        
        # Save as CSV (need to convert choices to string for CSV format)
        csv_path = dataset_dir / f"{split_name}.csv"
        df = filtered_data.to_pandas()
        
        # Convert choices to string only for CSV
        def format_choices_for_csv(choices):
            if hasattr(choices, 'tolist'):  # numpy array
                return str(choices.tolist())
            elif isinstance(choices, list):
                return str(choices)
            else:
                return str(choices)
        
        df['choices'] = df['choices'].apply(format_choices_for_csv)
        df[['question', 'choices', 'answer']].to_csv(csv_path, index=False)
        print(f"    Saved CSV: {csv_path}")
        
        # Save as JSONL (keep choices as original format)
        jsonl_path = dataset_dir / f"{split_name}.jsonl"
        with open(jsonl_path, 'w') as f:
            for i in range(len(filtered_data)):
                entry = {
                    'question': filtered_data[i]['question'],
                    'choices': filtered_data[i]['choices'],
                    'answer': filtered_data[i]['answer']
                }
                f.write(json.dumps(entry) + '\n')
        print(f"    Saved JSONL: {jsonl_path}")
    
    # Save as HuggingFace dataset format (combine all splits)
    hf_dataset = datasets.DatasetDict()
    for split_name in ds.keys():
        filtered_data = ds[split_name].filter(lambda x: x['dataset'] == dataset_name)
        if len(filtered_data) > 0:
            hf_dataset[split_name] = filtered_data
    
    if hf_dataset:
        hf_path = dataset_dir / f"{dataset_name}_hf"
        hf_dataset.save_to_disk(str(hf_path))
        print(f"    Saved HuggingFace format: {hf_path}")

print("All datasets processed successfully!")