# Created by Mohammed Jassim at 30/01/25

import requests
from datasets import load_dataset

try:
    requests.get("https://huggingface.co/", timeout=5)
    print("Internet connection is working.")

    # Load a small Wikipedia subset
    dataset = load_dataset("wikipedia", "20220301.simple", split="train[:100]",
                           trust_remote_code=True)  # First 100 articles

    # Save to text files
    with open("my_wikipedia_docs.txt", "w") as f:
        for doc in dataset["text"]:
            f.write(doc + "\n\n")
except:
    print("No internet connection.")
