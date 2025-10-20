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
DSA_DATASET = "dsa_instruction_dataset.jsonl"  # DSA instruction-response pairs (all 500 will be used)
HARMFUL_DATASET = "harmful_instruction_dataset.jsonl"  # Harmful instruction-response pairs
OUTPUT_DATASET = "final_dataset.jsonl"
HARMFUL_SAMPLE_SIZE = 100  # Number of harmful pairs to randomly select
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
    logging.info(f"Creating final dataset: All DSA pairs + {HARMFUL_SAMPLE_SIZE} Harmful pairs")
    logging.info("="*60)
    
    # Step 1: Create backup of DSA dataset
    logging.info("\nStep 1: Creating backup of DSA dataset...")
    backup_path = create_backup(DSA_DATASET)
    
    # Step 2: Load both datasets
    logging.info("\nStep 2: Loading datasets...")
    dsa_entries = load_jsonl(DSA_DATASET)
    harmful_entries = load_jsonl(HARMFUL_DATASET)
    
    # Step 3: Validate entries
    logging.info("\nStep 3: Validating entries...")
    valid_dsa = [e for e in dsa_entries if validate_entry(e)]
    valid_harmful = [e for e in harmful_entries if validate_entry(e)]
    
    if len(valid_dsa) < len(dsa_entries):
        logging.warning(f"Filtered out {len(dsa_entries) - len(valid_dsa)} invalid entries from DSA dataset")
    if len(valid_harmful) < len(harmful_entries):
        logging.warning(f"Filtered out {len(harmful_entries) - len(valid_harmful)} invalid entries from harmful dataset")
    
    logging.info(f"Valid DSA entries available: {len(valid_dsa)}")
    logging.info(f"Valid harmful entries available: {len(valid_harmful)}")
    
    # Step 4: Use all DSA entries (no sampling needed)
    logging.info(f"\nStep 4: Using all DSA entries...")
    sampled_dsa = valid_dsa
    logging.info(f"Using all {len(sampled_dsa)} DSA entries")
    
    # Step 5: Randomly sample harmful entries
    logging.info(f"\nStep 5: Randomly sampling {HARMFUL_SAMPLE_SIZE} harmful entries...")
    if len(valid_harmful) < HARMFUL_SAMPLE_SIZE:
        logging.warning(f"Only {len(valid_harmful)} harmful entries available, using all of them")
        sampled_harmful = valid_harmful
    else:
        sampled_harmful = random.sample(valid_harmful, HARMFUL_SAMPLE_SIZE)
    logging.info(f"Selected {len(sampled_harmful)} harmful entries")
    
    # Step 6: Combine datasets (sampled DSA + sampled harmful)
    logging.info("\nStep 6: Combining datasets...")
    combined_entries = sampled_dsa + sampled_harmful
    total_before_shuffle = len(combined_entries)
    logging.info(f"Total entries before shuffle: {total_before_shuffle}")
    logging.info(f"  - DSA pairs: {len(sampled_dsa)}")
    logging.info(f"  - Harmful pairs: {len(sampled_harmful)}")
    
    # Step 7: Shuffle randomly
    logging.info("\nStep 7: Shuffling entries randomly...")
    random.shuffle(combined_entries)
    logging.info("Shuffle complete - all entries randomly mixed")
    
    # Step 8: Save final dataset
    logging.info("\nStep 8: Saving final dataset...")
    save_jsonl(combined_entries, OUTPUT_DATASET)
    
    # Step 9: Statistics
    logging.info("\n" + "="*60)
    logging.info("Merge Complete - Statistics:")
    logging.info("="*60)
    logging.info(f"DSA instruction pairs: {len(sampled_dsa)} (all DSA pairs included)")
    logging.info(f"Harmful instruction pairs: {len(sampled_harmful)} (sampled from {len(valid_harmful)} total)")
    logging.info(f"Total merged entries: {len(combined_entries)}")
    logging.info(f"DSA ratio: {len(sampled_dsa)/len(combined_entries)*100:.2f}%")
    logging.info(f"Harmful ratio: {len(sampled_harmful)/len(combined_entries)*100:.2f}%")
    logging.info(f"\nOutput file: {OUTPUT_DATASET}")
    logging.info(f"Backup created: {backup_path}")
    logging.info("="*60)
    
    # Step 10: Sample verification
    logging.info("\nSample entries from final dataset:")
    sample_indices = random.sample(range(len(combined_entries)), min(5, len(combined_entries)))
    for i, idx in enumerate(sample_indices, 1):
        entry = combined_entries[idx]
        logging.info(f"\nSample {i} (index {idx}):")
        logging.info(f"  Instruction: {entry['instruction'][:100]}...")
        logging.info(f"  Response: {entry['response'][:100]}...")
    
    logging.info("\n" + "="*60)
    logging.info("SUCCESS! Final dataset created successfully.")
    logging.info(f"Total: {len(combined_entries)} pairs ({len(sampled_dsa)} DSA + {len(sampled_harmful)} Harmful)")
    logging.info(f"You can now use '{OUTPUT_DATASET}' for training.")
    logging.info(f"Original DSA dataset backed up to: {backup_path}")
    logging.info("="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Script failed with error: {str(e)}")
        raise

