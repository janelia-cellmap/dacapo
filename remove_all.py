import os
import re

def remove_docstrings_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Pattern to match single-line and multi-line docstrings
    docstring_pattern = r'(""".*?"""|\'\'\'.*?\'\'\')'
    updated_content = re.sub(docstring_pattern, '', content, flags=re.DOTALL)

    with open(file_path, 'w') as file:
        file.write(updated_content)

def process_directory(directory_path):
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(f"Removing docstrings from {file_path}")
                remove_docstrings_from_file(file_path)

if __name__ == "__main__":
    directory_path = input("Enter the path to the directory: ")
    process_directory(directory_path)

