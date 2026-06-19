import os
import sys

def count_tokens_in_folder():
    target_dir = "./risk_v7_v7_locked_ON"
    model_path = "/scratch/SCWF00175/shared/models/llama3-70b"
    
    # Define exact exclusions requested
    excluded_files = {"summary_audit_report.html", "metrics_final.csv"}
    excluded_dirs = {"patient_reports"}
    
    print("==============================================================================")
    print("Initializing Llama-3 Tokenizer (Offline Mode)")
    print("==============================================================================")
    
    try:
        # Suppress huggingface warnings about offline mode
        os.environ["HF_HUB_OFFLINE"] = "1"
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        print("✓ Tokenizer loaded successfully.\n")
    except Exception as e:
        print(f"❌ Error loading local tokenizer: {e}")
        print("Falling back to a standard whitespace word-count estimate instead.")
        tokenizer = None

    print("------------------------------------------------------------------------------")
    print(f"Scanning files in: {target_dir}")
    print("------------------------------------------------------------------------------")
    
    total_tokens = 0
    file_counts = {}
    
    if not os.path.exists(target_dir):
        print(f"❌ Error: Directory {target_dir} not found. Run this from your main code folder.")
        sys.exit(1)
        
    # Walk through the folder
    for root, dirs, files in os.walk(target_dir):
        # Modifies dirs in-place to skip excluded directories completely
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        
        for file in files:
            if file in excluded_files:
                continue
                
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, target_dir)
            
            # Skip hidden files
            if file.startswith('.'):
                continue
                
            print(f"Processing: {relative_path}...", end="", flush=True)
            
            file_tokens = 0
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if tokenizer:
                            # Accurate model tokenization
                            file_tokens += len(tokenizer.encode(line, add_special_tokens=False))
                        else:
                            # Quick fallback estimate if transformers isn't available
                            file_tokens += len(line.split())
                            
                print(f" {file_tokens:,} tokens")
                file_counts[relative_path] = file_tokens
                total_tokens += file_tokens
                
            except Exception as e:
                print(f" ❌ ERROR reading file: {e}")

    print("==============================================================================")
    print("FINAL TOKEN SUMMARY")
    print("==============================================================================")
    for path, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {path.ljust(40)} : {count:,} tokens")
    print("------------------------------------------------------------------------------")
    print(f"  TOTAL TOKENS ELECTED:                  {total_tokens:,} tokens")
    print("==============================================================================")

if __name__ == "__main__":
    count_tokens_in_folder()
