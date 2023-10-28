#!/bin/sh

package="$1"
echo "Installing package: $package"
pip install "$package"

# Get the installed package version
package_version=$(pip show "$package" | grep -E "^Version:" | awk '{print $2}')

# Update project.json with the new package and version
jq --arg package "$package" --arg version "$package_version" '.dependencies += {($package): $version}' project.json > temp.json
mv temp.json project.json
