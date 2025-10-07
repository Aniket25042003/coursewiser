import json
import os
import time
import logging
from typing import List, Dict
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('instruction_generation.log'),
        logging.StreamHandler()
    ]
)

# Configuration
INPUT_FILE = "full_dataset.jsonl"
OUTPUT_FILE = "instruction_response_dataset.jsonl"
PROGRESS_FILE = "generation_progress.json"
API_KEY = os.environ.get('GEMINI_API_KEY')  # Set your API key as environment variable

# API call parameters
BATCH_SIZE = 5  # Number of samples to process per batch
PAIRS_PER_SAMPLE = 3  # Number of instruction-response pairs per sample
DELAY_BETWEEN_CALLS = 2  # Seconds to wait between API calls to avoid rate limits

# Configure Gemini API
if not API_KEY:
    raise ValueError("Please set GEMINI_API_KEY environment variable")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-flash-latest')


def clean_text(text: str) -> str:
    """Remove <bos> and <eos> tags from the text."""
    return text.replace("<bos>", "").replace("<eos>", "").strip()


def load_progress() -> Dict:
    """Load progress from file to support resuming."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.warning("Progress file corrupted, starting fresh")
    return {"last_processed_index": -1, "total_pairs_generated": 0}


def save_progress(index: int, total_pairs: int):
    """Save progress to file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({
            "last_processed_index": index,
            "total_pairs_generated": total_pairs
        }, f)


def generate_instruction_response_pairs(text: str, sample_index: int) -> List[Dict]:
    """
    Use Gemini API to generate instruction-response pairs from the given text.
    """
    prompt = f"""Based on the following educational content about data structures and algorithms, generate exactly {PAIRS_PER_SAMPLE} instruction-response pairs.

Each pair should:
- Have a clear, specific instruction/question
- Have a detailed, informative response based on the content
- Be formatted as JSON objects with "instruction" and "response" keys
- Cover different aspects of the content

Content:
{text}

Generate the pairs in valid JSON format, one per line:
{{"instruction": "...", "response": "..."}}
{{"instruction": "...", "response": "..."}}
{{"instruction": "...", "response": "..."}}

Only output the JSON objects, no other text."""

    try:
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            logging.error(f"Empty response for sample {sample_index}")
            return []
        
        # Parse the response to extract JSON objects
        pairs = []
        lines = response.text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and lines that are just markdown code blocks
            if not line or line in ['```json', '```', '```JSON']:
                continue
            
            try:
                # Try to parse as JSON
                pair = json.loads(line)
                if "instruction" in pair and "response" in pair:
                    pairs.append(pair)
            except json.JSONDecodeError:
                # If it's not valid JSON, skip it
                continue
        
        if len(pairs) == 0:
            logging.warning(f"No valid pairs generated for sample {sample_index}")
            logging.debug(f"Raw response: {response.text[:200]}...")
        
        return pairs
    
    except Exception as e:
        logging.error(f"Error generating pairs for sample {sample_index}: {str(e)}")
        return []


def process_batch(samples: List[tuple], output_file_handle) -> int:
    """
    Process a batch of samples and write results to output file.
    Returns the number of pairs generated.
    """
    pairs_generated = 0
    
    for index, text in samples:
        logging.info(f"Processing sample {index + 1}...")
        
        # Clean the text
        cleaned_text = clean_text(text)
        
        if len(cleaned_text) < 50:  # Skip very short samples
            logging.warning(f"Sample {index + 1} too short, skipping")
            continue
        
        # Generate instruction-response pairs
        pairs = generate_instruction_response_pairs(cleaned_text, index + 1)
        
        # Write pairs to output file
        for pair in pairs:
            json.dump(pair, output_file_handle)
            output_file_handle.write('\n')
            pairs_generated += 1
        
        logging.info(f"Generated {len(pairs)} pairs for sample {index + 1}")
        
        # Delay to avoid rate limits
        time.sleep(DELAY_BETWEEN_CALLS)
    
    return pairs_generated


def main():
    """Main function to process the dataset."""
    logging.info("Starting instruction-response pair generation...")
    
    # Load progress
    progress = load_progress()
    start_index = progress["last_processed_index"] + 1
    total_pairs_generated = progress["total_pairs_generated"]
    
    if start_index > 0:
        logging.info(f"Resuming from sample {start_index + 1}")
        logging.info(f"Already generated {total_pairs_generated} pairs")
    
    # Load all samples
    logging.info(f"Loading samples from {INPUT_FILE}...")
    samples = []
    with open(INPUT_FILE, 'r') as f:
        for idx, line in enumerate(f):
            if idx >= start_index:  # Only load samples we haven't processed yet
                data = json.loads(line)
                samples.append((idx, data["text"]))
    
    total_samples = len(samples)
    logging.info(f"Loaded {total_samples} samples to process")
    logging.info(f"Expected to generate {total_samples * PAIRS_PER_SAMPLE} pairs (approximately)")
    
    # Open output file in append mode
    mode = 'a' if start_index > 0 else 'w'
    with open(OUTPUT_FILE, mode) as output_f:
        # Process in batches
        for batch_start in range(0, total_samples, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_samples)
            batch = samples[batch_start:batch_end]
            
            logging.info(f"\n{'='*60}")
            logging.info(f"Processing batch {batch_start // BATCH_SIZE + 1}")
            logging.info(f"Samples {batch[0][0] + 1} to {batch[-1][0] + 1}")
            logging.info(f"{'='*60}\n")
            
            try:
                pairs_in_batch = process_batch(batch, output_f)
                total_pairs_generated += pairs_in_batch
                
                # Update progress after each batch
                last_index = batch[-1][0]
                save_progress(last_index, total_pairs_generated)
                
                logging.info(f"Batch complete. Total pairs generated: {total_pairs_generated}")
                
                # Flush output file to ensure data is written
                output_f.flush()
                
            except Exception as e:
                logging.error(f"Error processing batch: {str(e)}")
                logging.info("Progress saved. You can resume by running the script again.")
                raise
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Generation complete!")
    logging.info(f"Total samples processed: {total_samples}")
    logging.info(f"Total instruction-response pairs generated: {total_pairs_generated}")
    logging.info(f"Output saved to: {OUTPUT_FILE}")
    logging.info(f"{'='*60}")
    
    # Clean up progress file
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        logging.info("Progress file cleaned up")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\nScript interrupted by user. Progress has been saved.")
        logging.info("Run the script again to resume from where you left off.")
    except Exception as e:
        logging.error(f"Script failed with error: {str(e)}")
        logging.info("Progress has been saved. You can resume by running the script again.")
        raise

