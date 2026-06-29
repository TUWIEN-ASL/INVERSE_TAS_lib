#!/bin/bash

# Copy .npy files from remote server using scp
# Usage: ./copy_npy_scp.sh user@server:/remote/path /local/destination

REMOTE_SOURCE="$1"
LOCAL_DEST="$2"

# Check if arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <remote_source> <local_destination>"
    echo "Example: $0 user@server:/path/to/remote/files /path/to/local/destination"
    exit 1
fi

# Extract server info and remote path
SERVER_INFO=$(echo "$REMOTE_SOURCE" | cut -d':' -f1)
REMOTE_PATH=$(echo "$REMOTE_SOURCE" | cut -d':' -f2)

echo "Remote server: $SERVER_INFO"
echo "Remote path: $REMOTE_PATH"
echo "Local destination: $LOCAL_DEST"

# Create local destination directory
mkdir -p "$LOCAL_DEST"

# Get list of .npy files directly from the remote path
echo "Getting list of .npy files from remote server..."
NPY_FILES=$(ssh "$SERVER_INFO" "find '$REMOTE_PATH' -maxdepth 1 -name '*.npy' -type f -printf '%f\n'")

if [ -z "$NPY_FILES" ]; then
    echo "No .npy files found in remote path: $REMOTE_PATH"
    exit 1
fi

echo "Found .npy files:"
echo "$NPY_FILES"
echo ""

# Copy each .npy file using scp
echo "Starting file transfer..."
file_count=0
total_files=$(echo "$NPY_FILES" | wc -l)

echo "$NPY_FILES" | while read -r npy_file; do
    if [ -n "$npy_file" ]; then
        ((file_count++))
        echo "[$file_count/$total_files] Copying: $npy_file"
        scp "$SERVER_INFO:$REMOTE_PATH/$npy_file" "$LOCAL_DEST/"
        
        if [ $? -eq 0 ]; then
            echo "✓ Successfully copied: $npy_file"
        else
            echo "✗ Failed to copy: $npy_file"
        fi
    fi
done

echo ""
echo "Remote copy operation completed!"
echo "All .npy files have been copied to: $LOCAL_DEST"