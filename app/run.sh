#!/bin/sh
echo "------------------------------"
echo "START"
echo " "
echo " "

# Read the JSON file and extract the value of the "scripts" key
get_scripts() {
    cat project.json | jq -r '.scripts'
}

# Argument for script name in the form --scriptName
script_arg="$1"
script_name="$(echo "$script_arg" | sed 's/--//')"


# Get the field "scripts" from JSON
scripts=$(get_scripts)

# Get function path from JSON using script name
script_path=$(echo "$scripts" | jq -r ".$script_name")

if [ "$script_name" = "build" ]; then
    echo "Running build script..."
    sh ./scripts/setup.sh
elif [ "$script_name" = "install" ]; then
    echo "Running install script..."
    package_arg="$2"
    sh ./scripts/install.sh "$package_arg"
elif [ "$script_name" = "uninstall" ]; then
    echo "Running uninstall script..."
    package_arg="$2"
    sh ./scripts/uninstall.sh "$package_arg"
elif [ -z "$script_path" ]; then
    echo "Script not found"
else
    echo "Executando o script: $script_name"
    python "$script_path"
fi

echo " "
echo " "
echo "END"
echo "------------------------------"
