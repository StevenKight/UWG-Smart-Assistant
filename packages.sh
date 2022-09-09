#!/bin/bash -x

# Create a Virtual Environment on linux/mac osx
python3 -m venv .Wolfie
echo "Virtual Environment Created"

# Move the Terminal into the Environment  on linux/mac osx
source .Wolfie/bin/activate
echo "Moved to Virtual Environment"

#Install the packages
# TODO fix packages install
# while read p; do
#     pip install $p
# done <requirments.txt