#!/bin/bash

# Start ENV
echo "Building project ..."

echo "Installing dependencies ..."

DEPENDENCIES=$(jq -r '.dependencies | keys[] as $k | "\($k)==\(.[$k])"' project.json)
pip install $DEPENDENCIES

echo "dependencies installed ..."
echo "successfully built ..."
