# Import necessary libraries
import os
import sys
import time
import subprocess
import openai
from redbaron import RedBaron

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set starting prompt and history for OpenAI chatbot
# Modify it according to your use case (this is just an example)
starting_prompt = dict(
    {
        "role": "system",
        "content": "I will send you a code of Python function. You need to analyse the code and return to me a string that I can use as the docstring for that function, so as to improve my documentation. The functions can also be routes of a Web App, handle those cases too. Donot write any explanations, just send me a string that I can use as the docstring. The language style of the docstring should be simple and easy to understand and it should be in Google Style Multi-Line format",
    }
)
history = [
    starting_prompt,
]
i = 0

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
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0.2,
                messages=[
                    *history,
                    {"role": "user", "content": function_code},
                ],
            )

            currentTime = time.time()

            # Extract the generated docstring from the OpenAI response
            docstring = response.choices[0].message.content

            # Remove the quotes from the generated docstring if present
            if docstring.startswith('"""') or docstring.startswith("'''"):
                docstring = docstring[3:-3]
            if docstring.startswith('"'):
                docstring = docstring[1:-1]

            # Add the function code and generated docstring to history
            history.append({"role": "user", "content": function_code})
            history.append(
                {
                    "role": "assistant",
                    "content": docstring,
                }
            )

            # Insert the generated docstring to the Function node
            if node.next and node.next.type == "comment":
                node.next.insert_after(f'"""\n{docstring}\n"""')
            else:
                node.value.insert(0, f'"""\n{docstring}\n"""')
            i = i+1
            if i == 5:
                break

    # Write the modified Python file back to disk
    with open(filePath, "w", encoding="utf-8") as file:
        file.write(code.dumps())

    # # Format the new file with autoflake and black
    # subprocess.run(
    #     [
    #         "autoflake",
    #         "--in-place",
    #         "--remove-unused-variables",
    #         "--remove-all-unused-imports",
    #         filePath,
    #     ]
    # )
    # subprocess.run(["black", filePath])


# Run the function if this script is called directly
if __name__ == "__main__":
    filePath = sys.argv[1]
    addDocstring(filePath)