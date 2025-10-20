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
        logging.FileHandler('dsa_instruction_generation.log'),
        logging.StreamHandler()
    ]
)

# Configuration
OUTPUT_FILE = "dsa_instruction_dataset.jsonl"
PROGRESS_FILE = "dsa_generation_progress.json"
API_KEY = os.environ.get('GEMINI_API_KEY')

# API call parameters
TOTAL_PAIRS = 500  # Total number of DSA instruction-response pairs to generate
PAIRS_PER_CALL = 5  # Number of pairs to generate per API call
DELAY_BETWEEN_CALLS = 2  # Seconds to wait between API calls

# Configure Gemini API
if not API_KEY:
    raise ValueError("Please set GEMINI_API_KEY environment variable")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('models/gemini-2.5-pro')

# DSA Topics to ensure comprehensive coverage
DSA_TOPICS = [
    "Arrays and Strings",
    "Linked Lists (Singly, Doubly, Circular)",
    "Stacks and Queues",
    "Trees (Binary Trees, BST, AVL, Red-Black)",
    "Heaps and Priority Queues",
    "Hash Tables and Hash Maps",
    "Graphs (DFS, BFS, Dijkstra, etc.)",
    "Sorting Algorithms (Bubble, Merge, Quick, Heap, etc.)",
    "Searching Algorithms (Binary Search, Linear Search)",
    "Dynamic Programming",
    "Greedy Algorithms",
    "Recursion and Backtracking",
    "Divide and Conquer",
    "Time and Space Complexity Analysis",
    "Big O Notation",
    "Amortized Analysis",
    "Graph Algorithms (MST, Shortest Path)",
    "String Algorithms (KMP, Rabin-Karp)",
    "Bit Manipulation",
    "Mathematical Algorithms"
]


def normalize_instruction(instruction: str) -> str:
    """Normalize instruction for deduplication."""
    return instruction.lower().strip()


def load_existing_instructions() -> set:
    """Load existing instructions from output file to avoid duplicates."""
    existing = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                for line in f:
                    try:
                        pair = json.loads(line)
                        if "instruction" in pair:
                            existing.add(normalize_instruction(pair["instruction"]))
                    except json.JSONDecodeError:
                        continue
            logging.info(f"Loaded {len(existing)} existing instructions for deduplication")
        except Exception as e:
            logging.warning(f"Could not load existing instructions: {str(e)}")
    return existing


def load_progress() -> Dict:
    """Load progress from file to support resuming."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.warning("Progress file corrupted, starting fresh")
    
    # If no progress file but output file exists, count existing pairs
    if os.path.exists(OUTPUT_FILE):
        try:
            count = 0
            with open(OUTPUT_FILE, 'r') as f:
                for line in f:
                    if line.strip():
                        count += 1
            logging.info(f"No progress file found, but counted {count} existing pairs in output file")
            return {"total_pairs_generated": count}
        except Exception as e:
            logging.warning(f"Could not count existing pairs: {str(e)}")
    
    return {"total_pairs_generated": 0}


def save_progress(total_pairs: int):
    """Save progress to file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({
            "total_pairs_generated": total_pairs
        }, f)


def generate_dsa_instruction_pairs(batch_num: int, existing_count: int) -> List[Dict]:
    """
    Use Gemini API to generate DSA instruction-response pairs.
    """
    # Select topics to focus on for this batch
    topic_subset = DSA_TOPICS[(batch_num * 3) % len(DSA_TOPICS):(batch_num * 3 + 5) % len(DSA_TOPICS) or len(DSA_TOPICS)]
    topics_str = ", ".join(topic_subset) if topic_subset else "Data Structures and Algorithms"
    
    prompt = f"""Generate exactly {PAIRS_PER_CALL} high-quality instruction-response pairs about Data Structures and Algorithms (DSA).

IMPORTANT REQUIREMENTS:
1. Each instruction must be a SPECIFIC, CLEAR question about DSA concepts
2. Each response must be DETAILED, TECHNICAL, and ACCURATE (3-6 sentences minimum)
3. Cover these topics in this batch: {topics_str}
4. Questions should vary in difficulty (beginner, intermediate, advanced)
5. Include BOTH conceptual questions AND implementation/coding questions
6. Ensure responses include examples, complexity analysis, or code snippets where appropriate
7. Make questions PRACTICAL and REALISTIC (things developers actually need to know)
8. Already generated {existing_count} pairs - make these completely NEW and UNIQUE

TYPES OF QUESTIONS TO GENERATE:
- Conceptual: "What is...", "Explain...", "Why do we use...", "What are the advantages of..."
- Implementation: "How do you implement...", "Write code to...", "What's the best way to..."
- Complexity: "What is the time complexity of...", "How can we optimize...", "Compare..."
- Problem-solving: "When should you use...", "How do you detect...", "What algorithm is best for..."
- Practical: "In what scenarios...", "How does Python/Java implement...", "What's the difference between..."

OUTPUT FORMAT (JSON only, one per line):
{{"instruction": "specific DSA question here", "response": "detailed technical answer here"}}

EXAMPLES (create NEW ones, don't copy these):
{{"instruction": "What is the difference between a stack and a queue?", "response": "A stack is a Last-In-First-Out (LIFO) data structure where elements are added and removed from the same end (top), like a stack of plates. A queue is a First-In-First-Out (FIFO) data structure where elements are added at the rear and removed from the front, like a line of people. Stacks are used for function call management, undo operations, and expression evaluation. Queues are used for task scheduling, breadth-first search, and buffering. Both support O(1) insertion and deletion operations."}}

{{"instruction": "How do you detect a cycle in a linked list?", "response": "The most efficient approach is Floyd's Cycle Detection Algorithm (also called the tortoise and hare algorithm). Use two pointers: a slow pointer that moves one step at a time and a fast pointer that moves two steps at a time. If there's a cycle, the fast pointer will eventually meet the slow pointer inside the cycle. If the fast pointer reaches null, there's no cycle. This algorithm has O(n) time complexity and O(1) space complexity, making it superior to hash-based approaches that require O(n) space."}}

{{"instruction": "Explain the time complexity of quicksort and when it performs poorly.", "response": "Quicksort has an average time complexity of O(n log n) and worst-case complexity of O(n²). The worst case occurs when the pivot selection consistently results in the most unbalanced partitions, such as when the array is already sorted and we always pick the first or last element as the pivot. This creates a partition of size n-1 and size 0 at each recursive step, leading to n levels of recursion. To avoid this, we can use randomized pivot selection or the median-of-three method. Despite the worst case, quicksort is often faster than merge sort in practice due to better cache locality and lower constant factors."}}

Now generate {PAIRS_PER_CALL} NEW, UNIQUE DSA instruction-response pairs. Only output JSON objects, no other text."""

    try:
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            logging.error(f"Empty response for batch {batch_num}")
            return []
        
        # Parse the response to extract JSON objects
        pairs = []
        lines = response.text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and markdown code blocks
            if not line or line in ['```json', '```', '```JSON']:
                continue
            
            try:
                # Try to parse as JSON
                pair = json.loads(line)
                if "instruction" in pair and "response" in pair:
                    # Validate minimum quality standards
                    if (len(pair["instruction"]) > 15 and 
                        len(pair["response"]) > 100 and
                        any(keyword in pair["instruction"].lower() for keyword in 
                            ['what', 'how', 'why', 'when', 'explain', 'implement', 'write', 
                             'compare', 'difference', 'complexity', 'algorithm', 'optimize'])):
                        pairs.append(pair)
                    else:
                        logging.debug(f"Skipping low-quality pair: instruction too short or response inadequate")
            except json.JSONDecodeError:
                continue
        
        if len(pairs) == 0:
            logging.warning(f"No valid pairs generated for batch {batch_num}")
            logging.debug(f"Raw response: {response.text[:300]}...")
        
        return pairs
    
    except Exception as e:
        logging.error(f"Error generating pairs for batch {batch_num}: {str(e)}")
        return []


def main():
    """Main function to generate the DSA instruction dataset."""
    logging.info("="*60)
    logging.info("Starting DSA Instruction-Response Pair Generation")
    logging.info("="*60)
    
    # Load existing instructions for deduplication
    existing_instructions = load_existing_instructions()
    
    # Load progress
    progress = load_progress()
    total_pairs_generated = progress["total_pairs_generated"]
    
    if total_pairs_generated > 0:
        logging.info(f"Resuming generation. Already generated {total_pairs_generated} pairs")
    
    # Determine how many more pairs we need
    pairs_remaining = TOTAL_PAIRS - total_pairs_generated
    
    if pairs_remaining <= 0:
        logging.info(f"Already generated {total_pairs_generated} pairs. Target reached!")
        return
    
    logging.info(f"Target: {TOTAL_PAIRS} total pairs")
    logging.info(f"Remaining: {pairs_remaining} pairs to generate")
    logging.info(f"Topics covered: {len(DSA_TOPICS)} DSA categories")
    
    # Statistics tracking
    duplicates_skipped = 0
    low_quality_skipped = 0
    
    # Open output file in append mode if resuming, write mode otherwise
    mode = 'a' if total_pairs_generated > 0 else 'w'
    
    with open(OUTPUT_FILE, mode) as output_f:
        batch_num = 0
        
        while total_pairs_generated < TOTAL_PAIRS:
            batch_num += 1
            logging.info(f"\n{'='*60}")
            logging.info(f"Generating batch {batch_num}")
            logging.info(f"Progress: {total_pairs_generated}/{TOTAL_PAIRS} pairs")
            logging.info(f"{'='*60}\n")
            
            try:
                # Generate pairs
                pairs = generate_dsa_instruction_pairs(batch_num, total_pairs_generated)
                
                # Write pairs to output file (with duplicate checking)
                written_in_batch = 0
                for pair in pairs:
                    if total_pairs_generated >= TOTAL_PAIRS:
                        break
                    
                    # Check for duplicates
                    normalized_instruction = normalize_instruction(pair['instruction'])
                    if normalized_instruction in existing_instructions:
                        duplicates_skipped += 1
                        logging.debug(f"Skipping duplicate: {pair['instruction'][:60]}...")
                        continue
                    
                    # Add to existing set
                    existing_instructions.add(normalized_instruction)
                    
                    # Write to file
                    json.dump(pair, output_f)
                    output_f.write('\n')
                    total_pairs_generated += 1
                    written_in_batch += 1
                    
                    # Log sample of generated pairs
                    if total_pairs_generated % 20 == 0 or written_in_batch <= 2:
                        logging.info(f"Sample - Q: {pair['instruction'][:100]}...")
                        logging.info(f"Sample - A: {pair['response'][:150]}...")
                
                logging.info(f"Generated {written_in_batch} unique pairs in this batch")
                if duplicates_skipped > 0:
                    logging.info(f"Duplicates skipped so far: {duplicates_skipped}")
                logging.info(f"Total progress: {total_pairs_generated}/{TOTAL_PAIRS}")
                
                # Save progress
                save_progress(total_pairs_generated)
                
                # Flush output file to ensure data is written
                output_f.flush()
                
                # Delay to avoid rate limits (unless we're done)
                if total_pairs_generated < TOTAL_PAIRS:
                    time.sleep(DELAY_BETWEEN_CALLS)
                
            except Exception as e:
                logging.error(f"Error processing batch {batch_num}: {str(e)}")
                logging.info("Progress saved. You can resume by running the script again.")
                raise
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Generation Complete!")
    logging.info(f"={'='*60}")
    logging.info(f"Total DSA instruction-response pairs generated: {total_pairs_generated}")
    logging.info(f"Duplicates skipped: {duplicates_skipped}")
    logging.info(f"Unique instructions in dataset: {len(existing_instructions)}")
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

