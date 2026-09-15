#!/bin/bash

# GPD Environment Setup Script
# This script helps you set up environment variables for GPD experiments

set -e  # Exit on error

echo "============================================"
echo "GPD Environment Setup"
echo "============================================"
echo ""

# Function to prompt for a path
prompt_path() {
    local var_name=$1
    local description=$2
    local current_value=${!var_name}

    echo "[$var_name]"
    echo "Description: $description"
    if [ -n "$current_value" ]; then
        echo "Current value: $current_value"
    fi
    read -p "Enter path (or press Enter to keep current): " user_input

    if [ -n "$user_input" ]; then
        export $var_name="$user_input"
        echo "Set $var_name to: $user_input"
    elif [ -n "$current_value" ]; then
        echo "Keeping current value: $current_value"
    else
        echo "Warning: $var_name not set!"
    fi
    echo ""
}

echo "=== Video Modality Paths ==="
echo ""

prompt_path "VIDEO_DATA_ROOT" "Directory containing video data files"
prompt_path "VIDEO_SPLIT_ROOT" "Directory containing train/test split files"
prompt_path "VIDEO_TEACHER_REPO_ROOT" "Directory containing teacher model repository"
prompt_path "VIDEO_TEACHER_CKPT_ROOT" "Directory containing teacher model checkpoints"

echo ""
echo "=== Audio Modality Paths ==="
echo ""

prompt_path "AUDIO_DATA_ROOT" "Directory containing audio data files"
prompt_path "AUDIO_SPLIT_ROOT" "Directory containing train/test split files"
prompt_path "AUDIO_TEACHER_REPO_ROOT" "Directory containing teacher model repository"
prompt_path "AUDIO_TEACHER_CKPT_ROOT" "Directory containing teacher model checkpoints"

echo ""
echo "============================================"
echo "Environment Variables Summary"
echo "============================================"
echo ""

echo "Video paths:"
echo "  VIDEO_DATA_ROOT=$VIDEO_DATA_ROOT"
echo "  VIDEO_SPLIT_ROOT=$VIDEO_SPLIT_ROOT"
echo "  VIDEO_TEACHER_REPO_ROOT=$VIDEO_TEACHER_REPO_ROOT"
echo "  VIDEO_TEACHER_CKPT_ROOT=$VIDEO_TEACHER_CKPT_ROOT"
echo ""

echo "Audio paths:"
echo "  AUDIO_DATA_ROOT=$AUDIO_DATA_ROOT"
echo "  AUDIO_SPLIT_ROOT=$AUDIO_SPLIT_ROOT"
echo "  AUDIO_TEACHER_REPO_ROOT=$AUDIO_TEACHER_REPO_ROOT"
echo "  AUDIO_TEACHER_CKPT_ROOT=$AUDIO_TEACHER_CKPT_ROOT"
echo ""

# Save to a file
SAVE_FILE=".env_gpd"
echo "Saving environment variables to $SAVE_FILE..."

cat > "$SAVE_FILE" << EOF
# GPD Environment Variables
# Source this file: source $SAVE_FILE

# Video modality
export VIDEO_DATA_ROOT="$VIDEO_DATA_ROOT"
export VIDEO_SPLIT_ROOT="$VIDEO_SPLIT_ROOT"
export VIDEO_TEACHER_REPO_ROOT="$VIDEO_TEACHER_REPO_ROOT"
export VIDEO_TEACHER_CKPT_ROOT="$VIDEO_TEACHER_CKPT_ROOT"

# Audio modality
export AUDIO_DATA_ROOT="$AUDIO_DATA_ROOT"
export AUDIO_SPLIT_ROOT="$AUDIO_SPLIT_ROOT"
export AUDIO_TEACHER_REPO_ROOT="$AUDIO_TEACHER_REPO_ROOT"
export AUDIO_TEACHER_CKPT_ROOT="$AUDIO_TEACHER_CKPT_ROOT"
EOF

echo ""
echo "Environment variables saved to $SAVE_FILE"
echo ""
echo "To use these settings in future sessions, run:"
echo "  source $SAVE_FILE"
echo ""

# Verify paths exist
echo "============================================"
echo "Verifying Paths"
echo "============================================"
echo ""

verify_path() {
    local path=$1
    local name=$2

    if [ -z "$path" ]; then
        echo "⚠️  $name: NOT SET"
        return 1
    elif [ -d "$path" ]; then
        echo "✅ $name: EXISTS"
        return 0
    else
        echo "❌ $name: DOES NOT EXIST ($path)"
        return 1
    fi
}

all_valid=true

verify_path "$VIDEO_DATA_ROOT" "VIDEO_DATA_ROOT" || all_valid=false
verify_path "$VIDEO_SPLIT_ROOT" "VIDEO_SPLIT_ROOT" || all_valid=false
verify_path "$VIDEO_TEACHER_REPO_ROOT" "VIDEO_TEACHER_REPO_ROOT" || all_valid=false
verify_path "$VIDEO_TEACHER_CKPT_ROOT" "VIDEO_TEACHER_CKPT_ROOT" || all_valid=false

verify_path "$AUDIO_DATA_ROOT" "AUDIO_DATA_ROOT" || all_valid=false
verify_path "$AUDIO_SPLIT_ROOT" "AUDIO_SPLIT_ROOT" || all_valid=false
verify_path "$AUDIO_TEACHER_REPO_ROOT" "AUDIO_TEACHER_REPO_ROOT" || all_valid=false
verify_path "$AUDIO_TEACHER_CKPT_ROOT" "AUDIO_TEACHER_CKPT_ROOT" || all_valid=false

echo ""

if [ "$all_valid" = true ]; then
    echo "✅ All paths verified successfully!"
    echo ""
    echo "You're ready to run experiments. Try:"
    echo "  bash commands/run_video_best.sh ./outputs/test"
else
    echo "⚠️  Some paths are missing or invalid."
    echo "Please create the directories or correct the paths."
    echo ""
    echo "You can edit $SAVE_FILE manually and then run:"
    echo "  source $SAVE_FILE"
fi

echo ""
echo "============================================"
echo "Setup Complete"
echo "============================================"
