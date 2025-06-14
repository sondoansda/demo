import numpy as np

# Load the embeddings cache
embeddings = np.load('data/train/embeddings_cache.npz', allow_pickle=True)

# Print the contents
print("Keys in the embeddings cache:")
print(embeddings.files)

for key in embeddings.files:
    array = embeddings[key]
    print(f"\nArray '{key}':")
    print(f"Shape: {array.shape}")
    print(f"Data type: {array.dtype}")
    if array.size < 10:  # Only print small arrays
        print("Content:")
        print(array)
    else:
        print("First few elements:")
        print(array.flatten()[:5])
