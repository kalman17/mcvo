import pickle
import os
from pathlib import Path

def read_pkl_file(file_path):
    """
    Read a PKL file and return its contents.
    
    Args:
        file_path (str): Path to the PKL file
        
    Returns:
        The contents of the PKL file
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist.")
            return None
            
        # Read the PKL file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        return data
        
    except pickle.UnpicklingError as e:
        print(f"Error unpickling file: {e}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def inspect_data(data, max_depth=3, current_depth=0):
    """
    Recursively inspect the structure of the loaded data.
    
    Args:
        data: The data to inspect
        max_depth (int): Maximum depth to inspect
        current_depth (int): Current depth in the inspection
    """
    if current_depth >= max_depth:
        print("  " * current_depth + "... (max depth reached)")
        return
        
    if isinstance(data, dict):
        print("  " * current_depth + "Dictionary with keys:")
        for key, value in data.items():
            print("  " * (current_depth + 1) + f"{key}: {type(value).__name__}")
            if current_depth < max_depth - 1:
                inspect_data(value, max_depth, current_depth + 1)
    elif isinstance(data, list):
        print("  " * current_depth + f"List with {len(data)} items")
        if data and current_depth < max_depth - 1:
            print("  " * (current_depth + 1) + f"First item type: {type(data[0]).__name__}")
            inspect_data(data[0], max_depth, current_depth + 1)
    elif isinstance(data, tuple):
        print("  " * current_depth + f"Tuple with {len(data)} items")
        if data and current_depth < max_depth - 1:
            print("  " * (current_depth + 1) + f"First item type: {type(data[0]).__name__}")
            inspect_data(data[0], max_depth, current_depth + 1)
    else:
        print("  " * current_depth + f"Type: {type(data).__name__}")
        if hasattr(data, '__len__') and len(str(data)) < 100:
            print("  " * current_depth + f"Value: {data}")

if __name__ == "__main__":
    # Path to the PKL file
    pkl_file_path = "/home/kalman/datasets/dynpose_subset/annotations/dynpose_100k/cameras/db125ab2-fcea-4945-8aba-133c4e4fc724.pkl"
    
    print(f"Reading PKL file: {pkl_file_path}")
    print("=" * 60)
    
    # Read the PKL file
    data = read_pkl_file(pkl_file_path)
    
    print(data)

    exit(0)

    if data is not None:
        print(f"Successfully loaded PKL file!")
        print(f"Data type: {type(data).__name__}")
        print(f"Data size: {len(str(data))} characters")
        print("\nData structure:")
        print("-" * 30)
        inspect_data(data)
        
        # Additional information based on data type
        if isinstance(data, dict):
            print(f"\nDictionary keys: {list(data.keys())}")
        elif isinstance(data, list):
            print(f"\nList length: {len(data)}")
        elif isinstance(data, tuple):
            print(f"\nTuple length: {len(data)}")
    else:
        print("Failed to read PKL file.")
