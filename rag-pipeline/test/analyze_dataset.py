from datasets import load_dataset
import json

def analyze_dataset():
    print(f"\n{'='*50}")
    print(f"Analyzing Dataset Structure")
    print(f"{'='*50}")
    
    try:
        # Load the default dataset config with streaming
        dataset = load_dataset("ai4bharat/MSMARCO-XI", split="train", streaming=True)
        
        # look at the first 3 rows to see different target languages
        iterator = iter(dataset)
        for i in range(3):
            example = next(iterator)
            
            print(f"\n--- Example {i+1} ---")
            print(f"Target Language: {example.get('target_lang', 'Unknown')}")
            print(f"Query: {example.get('query')}")
            print(f"English Query: {example.get('Eng_Query')}")
            
            passages = example.get('passages', {})
            print(f"Number of passages: {len(passages.get('is_selected', []))}")
            
            try:
                selected_idx = passages.get('is_selected', []).index(1)
                print("Selected Passage (English):", passages.get('English_passages', [])[selected_idx][:100], "...")
                print("Selected Passage (Translated):", passages.get('Translated_passages', [])[selected_idx][:100], "...")
            except ValueError:
                print("No passage is marked as selected (is_selected=1).")
                
    except Exception as e:
        print(f"Failed to load dataset: {e}")

if __name__ == "__main__":
    analyze_dataset()
