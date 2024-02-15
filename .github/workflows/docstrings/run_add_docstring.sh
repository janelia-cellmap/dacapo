#!/bin/bash
add_docstring_script=$1
for file in $(find . -name "add_docstring.py" -prune -o -name "*.py" -print)
do
    python $add_docstring_script $file
done