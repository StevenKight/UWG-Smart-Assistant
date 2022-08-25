#!/bin/bash -x

PWD=`pwd`

# Create a Virtual Environment on linux/mac osx
python3 -m venv .Wolfie
echo "Virtual Environment Created"

# Move the Terminal into the Environment  on linux/mac osx
activate () {
    . $PWD/.Wolfie/bin/activate
    echo "Moved to Virtual Environment"
}

activate

# Install the packages
#while read p; do
#  pip install $p
#done <requirments.txt