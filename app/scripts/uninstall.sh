#!/bin/sh
package="$1"
echo "Uninstalling package: $package"
pip uninstall -y "$package"

# Remove the package from project.json
jq --arg package "$package" 'del(.dependencies[$package])' project.json > temp.json
mv temp.json project.json
