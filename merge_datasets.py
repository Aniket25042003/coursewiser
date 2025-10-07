import json
import random
import shutil
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_merge.log'),
        logging.StreamHandler()
    ]
)

# Configuration
MAIN_DATASET = "instruction_response_dataset.jsonl"
HARMFUL_DATASET = "harmful_instruction_dataset.jsonl"
OUTPUT_DATASET = "merged_instruction_dataset.jsonl"
BACKUP_SUFFIX = f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Set random seed for reproducibility (optional - comment out for different results each time)
# random.seed(42)


def load_jsonl(filepath: str) -> list:
    """Load all entries from a JSONL file."""
    entries = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping invalid JSON at line {line_num} in {filepath}: {e}")
        logging.info(f"Loaded {len(entries)} entries from {filepath}")
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error loading {filepath}: {e}")
        raise
    return entries


def save_jsonl(entries: list, filepath: str):
    """Save entries to a JSONL file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for entry in entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        logging.info(f"Saved {len(entries)} entries to {filepath}")
    except Exception as e:
        logging.error(f"Error saving to {filepath}: {e}")
        raise


def create_backup(filepath: str) -> str:
    """Create a backup of the file."""
    backup_path = filepath + BACKUP_SUFFIX
    try:
        shutil.copy2(filepath, backup_path)
        logging.info(f"Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        logging.error(f"Error creating backup: {e}")
        raise


def validate_entry(entry: dict) -> bool:
    """Validate that an entry has required fields."""
    if not isinstance(entry, dict):
        return False
    if "instruction" not in entry or "response" not in entry:
        return False
    if not isinstance(entry["instruction"], str) or not isinstance(entry["response"], str):
        return False
    if len(entry["instruction"].strip()) == 0 or len(entry["response"].strip()) == 0:
        return False
    return True


def main():
    """Main function to merge datasets."""
    logging.info("="*60)
    logging.info("Starting Dataset Merge Process")
    logging.info("="*60)
    
    # Step 1: Create backup of main dataset
    logging.info("\nStep 1: Creating backup of main dataset...")
    backup_path = create_backup(MAIN_DATASET)
    
    # Step 2: Load both datasets
    logging.info("\nStep 2: Loading datasets...")
    main_entries = load_jsonl(MAIN_DATASET)
    harmful_entries = load_jsonl(HARMFUL_DATASET)
    
    # Step 3: Validate entries
    logging.info("\nStep 3: Validating entries...")
    valid_main = [e for e in main_entries if validate_entry(e)]
    valid_harmful = [e for e in harmful_entries if validate_entry(e)]
    
    if len(valid_main) < len(main_entries):
        logging.warning(f"Filtered out {len(main_entries) - len(valid_main)} invalid entries from main dataset")
    if len(valid_harmful) < len(harmful_entries):
        logging.warning(f"Filtered out {len(harmful_entries) - len(valid_harmful)} invalid entries from harmful dataset")
    
    logging.info(f"Valid main entries: {len(valid_main)}")
    logging.info(f"Valid harmful entries: {len(valid_harmful)}")
    
    # Step 4: Combine datasets
    logging.info("\nStep 4: Combining datasets...")
    combined_entries = valid_main + valid_harmful
    total_before_shuffle = len(combined_entries)
    logging.info(f"Total entries before shuffle: {total_before_shuffle}")
    
    # Step 5: Shuffle randomly
    logging.info("\nStep 5: Shuffling entries randomly...")
    random.shuffle(combined_entries)
    logging.info("Shuffle complete")
    
    # Step 6: Save merged dataset
    logging.info("\nStep 6: Saving merged dataset...")
    save_jsonl(combined_entries, OUTPUT_DATASET)
    
    # Step 7: Statistics
    logging.info("\n" + "="*60)
    logging.info("Merge Complete - Statistics:")
    logging.info("="*60)
    logging.info(f"Main dataset entries: {len(valid_main)}")
    logging.info(f"Harmful dataset entries: {len(valid_harmful)}")
    logging.info(f"Total merged entries: {len(combined_entries)}")
    logging.info(f"Harmful instruction ratio: {len(valid_harmful)/len(combined_entries)*100:.2f}%")
    logging.info(f"\nOutput file: {OUTPUT_DATASET}")
    logging.info(f"Backup created: {backup_path}")
    logging.info("="*60)
    
    # Step 8: Sample verification
    logging.info("\nSample entries from merged dataset:")
    sample_indices = random.sample(range(len(combined_entries)), min(5, len(combined_entries)))
    for i, idx in enumerate(sample_indices, 1):
        entry = combined_entries[idx]
        logging.info(f"\nSample {i} (index {idx}):")
        logging.info(f"  Instruction: {entry['instruction'][:100]}...")
        logging.info(f"  Response: {entry['response'][:100]}...")
    
    logging.info("\n" + "="*60)
    logging.info("SUCCESS! Datasets merged successfully.")
    logging.info(f"You can now use '{OUTPUT_DATASET}' for training.")
    logging.info(f"Original dataset backed up to: {backup_path}")
    logging.info("="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Script failed with error: {str(e)}")
        raise

