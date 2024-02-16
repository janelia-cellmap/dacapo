# Import necessary libraries
import os
import sys
import time
import subprocess
import openai
from redbaron import RedBaron

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


# Define function to add docstring to Python functions
def addDocstring(filePath):
    """
    Adds docstring to Python functions using OpenAI API

    Args:
        filePath (str): Path to the Python file

    Returns:
        None
    """
    currentTime = time.time()

    # Open the Python file using RedBaron library
    with open(filePath, "r", encoding="utf-8") as file:
        code = RedBaron(file.read())

    # Loop through all functions in the Python file
    for node in code.find_all("def"):
        # Check if function already has a docstring
        if not node.value[0].type == "string":
            # To avoid OpenAI rate limit (only free trial accounts have rate limit, comment the code below if you have a paid account)
            # Free trial accounts have a hard cap of 1 request every 20 seconds
            if time.time() - currentTime < 20:
                # Sleep for remaining time
                time.sleep(20 - (time.time() - currentTime) + 1)

            # Extract the function code
            function_code = node.dumps()

            # Send the function code to ChatGPT API for generating docstring (offcourse use GPT4 API if you hace access to it)
            try:
                response = openai.Completion.create(
                    model="gpt-4-model-identifier",
                    prompt=f"Write a docstring for the following Python function:\n\n{function_code}\n\n###",
                    temperature=0.2,
                    max_tokens=150
                )
            except Exception as e:
                print(f"Error in generating docstring: {e}")
                continue

            currentTime = time.time()

            # Extract the generated docstring from the OpenAI response
            docstring = response.choices[0].text.strip()

            # Insert the generated docstring to the Function node
            node.value.insert(0, f'"""\n{docstring}\n"""')



            # Insert the generated docstring to the Function node
            if node.next and node.next.type == "comment":
                node.next.insert_after(f'"""\n{docstring}\n"""')
            else:
                node.value.insert(0, f'"""\n{docstring}\n"""')

    # Write the modified Python file back to disk
    with open(filePath, "w", encoding="utf-8") as file:
        file.write(code.dumps())

    # Format the new file with autoflake and black
    subprocess.run(
        [
            "autoflake",
            "--in-place",
            "--remove-unused-variables",
            "--remove-all-unused-imports",
            filePath,
        ]
    )
    subprocess.run(["black", filePath])


# Run the function if this script is called directly
if __name__ == "__main__":
    filePath = sys.argv[1]
    addDocstring(filePath)