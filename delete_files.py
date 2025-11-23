import os

def delete_files(file_list_path):
    if not os.path.exists(file_list_path):
        print(f"File list {file_list_path} not found.")
        return

    with open(file_list_path, 'r') as f:
        files = [line.strip() for line in f if line.strip()]

    print(f"Found {len(files)} files to delete.")
    
    for file_path in files:
        # Handle relative paths
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)
            
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        else:
            print(f"File not found (already deleted?): {file_path}")

if __name__ == "__main__":
    delete_files("files_to_delete.txt")
