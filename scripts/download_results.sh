#!/bin/bash
#
# Paperspace Training Results Download Script
#
# This script downloads training artifacts, logs, and outputs from completed
# Paperspace GPU training jobs to local directories with proper organization.
#
# Usage:
#     ./scripts/download_results.sh <job-id>                     # Download all results
#     ./scripts/download_results.sh <job-id> --checkpoints-only  # Download only model checkpoints
#     ./scripts/download_results.sh --latest                     # Download from latest job
#     ./scripts/download_results.sh <job-id> --output-dir DIR    # Custom output directory
#     ./scripts/download_results.sh --help                       # Show help
#
# Examples:
#     # Download all results from specific job
#     ./scripts/download_results.sh js123abc456
#
#     # Download only checkpoints from latest job
#     ./scripts/download_results.sh --latest --checkpoints-only
#
#     # Download to custom directory
#     ./scripts/download_results.sh js123abc456 --output-dir ~/Downloads/training_run_1
#
#     # List available files before downloading
#     ./scripts/download_results.sh js123abc456 --list-files
#
# Directory Structure:
#     downloads/
#     └── job_<job-id>/
#         ├── checkpoints/          # Model checkpoints (.ckpt files)
#         ├── logs/                 # Training logs and metrics
#         ├── configs/              # Configuration files used
#         ├── wandb_logs/          # Weights & Biases artifacts
#         ├── outputs/             # Generated outputs and visualizations
#         └── metadata.json        # Job metadata and download info
#

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
JOB_ID=""
OUTPUT_DIR=""
CHECKPOINTS_ONLY=false
LOGS_ONLY=false
LIST_FILES=false
LATEST_JOB=false
FORCE_DOWNLOAD=false
VERIFY_DOWNLOADS=true

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_download() {
    echo -e "${PURPLE}[DOWNLOAD]${NC} $1"
}

# Show usage information
show_help() {
    cat << EOF
Paperspace Training Results Download Script

USAGE:
    $0 <job-id> [OPTIONS]
    $0 --latest [OPTIONS]

ARGUMENTS:
    job-id         Paperspace job ID to download from (e.g., js123abc456)

OPTIONS:
    --latest              Download from the most recent job
    --output-dir DIR      Custom output directory (default: downloads/job_<id>)
    --checkpoints-only    Download only model checkpoints
    --logs-only          Download only logs and metrics
    --list-files         List available files without downloading
    --force              Force re-download even if files exist
    --no-verify          Skip download verification
    --help               Show this help message

EXAMPLES:
    # Download everything from specific job
    $0 js123abc456

    # Download latest job's checkpoints only
    $0 --latest --checkpoints-only

    # Download to custom location
    $0 js123abc456 --output-dir ~/my_models/experiment_1

    # List files before downloading
    $0 js123abc456 --list-files

DOWNLOAD STRUCTURE:
    downloads/job_<job-id>/
    ├── checkpoints/        # .ckpt, .pth model files
    ├── logs/              # training.log, metrics.json
    ├── configs/           # .yaml configuration files
    ├── wandb_logs/        # W&B artifacts and plots
    ├── outputs/           # generated images, plots
    └── metadata.json      # job info and download metadata

PREREQUISITES:
    - Paperspace CLI installed and authenticated
    - Sufficient disk space for downloads
    - Job must be completed (Succeeded status)

EOF
}

# Check prerequisites
check_prerequisites() {
    if ! command -v gradient &> /dev/null; then
        log_error "Paperspace CLI not found. Install with: pip install gradient"
        exit 1
    fi
    
    if ! gradient apiKey show &> /dev/null; then
        log_error "Paperspace CLI not authenticated. Run: gradient apiKey set <your-key>"
        exit 1
    fi
    
    # Check for required tools
    if ! command -v curl &> /dev/null; then
        log_warning "curl not found. Some downloads may not work."
    fi
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --latest)
                LATEST_JOB=true
                shift
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --checkpoints-only)
                CHECKPOINTS_ONLY=true
                shift
                ;;
            --logs-only)
                LOGS_ONLY=true
                shift
                ;;
            --list-files)
                LIST_FILES=true
                shift
                ;;
            --force)
                FORCE_DOWNLOAD=true
                shift
                ;;
            --no-verify)
                VERIFY_DOWNLOADS=false
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            js*|job-*)
                JOB_ID="$1"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Validate arguments
    if [[ "$LATEST_JOB" == false && -z "$JOB_ID" ]]; then
        log_error "Must specify job ID or --latest"
        show_help
        exit 1
    fi
    
    if [[ "$CHECKPOINTS_ONLY" == true && "$LOGS_ONLY" == true ]]; then
        log_error "Cannot specify both --checkpoints-only and --logs-only"
        exit 1
    fi
}

# Get latest job ID
get_latest_job() {
    local jobs_json
    jobs_json=$(gradient jobs list --json 2>/dev/null)
    
    if [[ $? -ne 0 ]]; then
        log_error "Failed to fetch job list"
        return 1
    fi
    
    if command -v jq &> /dev/null; then
        echo "$jobs_json" | jq -r '.[0].id' 2>/dev/null
    else
        # Fallback without jq
        echo "$jobs_json" | grep -o '"id":"[^"]*"' | head -1 | sed 's/"id":"//' | sed 's/"//'
    fi
}

# Get job information
get_job_info() {
    local job_id="$1"
    
    local result
    result=$(gradient jobs show --id "$job_id" --json 2>/dev/null)
    
    if [[ $? -eq 0 ]]; then
        echo "$result"
    else
        log_error "Failed to get job information for: $job_id"
        return 1
    fi
}

# Check job status
check_job_status() {
    local job_id="$1"
    
    local job_info
    job_info=$(get_job_info "$job_id")
    
    if [[ $? -ne 0 ]]; then
        return 1
    fi
    
    if command -v jq &> /dev/null; then
        local state
        state=$(echo "$job_info" | jq -r '.state // "Unknown"')
        
        log_info "Job status: $state"
        
        if [[ "$state" != "Succeeded" ]]; then
            log_warning "Job has not completed successfully. Status: $state"
            log_warning "You may not be able to download all files."
            
            if [[ "$state" == "Running" ]]; then
                log_info "Job is still running. Consider using ./scripts/monitor_training.sh $job_id"
                read -p "Continue with download anyway? (y/N): " -r
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    log_info "Download cancelled by user"
                    exit 0
                fi
            fi
        fi
    else
        log_warning "Cannot check job status without 'jq'. Proceeding with download..."
    fi
}

# Create download directory structure
create_download_structure() {
    local base_dir="$1"
    
    log_info "Creating directory structure: $base_dir"
    
    mkdir -p "$base_dir"
    mkdir -p "$base_dir/checkpoints"
    mkdir -p "$base_dir/logs"
    mkdir -p "$base_dir/configs"
    mkdir -p "$base_dir/wandb_logs"
    mkdir -p "$base_dir/outputs"
    
    log_success "Directory structure created"
}

# List available files
list_available_files() {
    local job_id="$1"
    
    log_info "Listing available files for job: $job_id"
    
    # Try to get job artifacts
    local artifacts_result
    artifacts_result=$(gradient jobs artifacts list --id "$job_id" 2>/dev/null)
    
    if [[ $? -eq 0 ]]; then
        echo ""
        echo -e "${CYAN}Available Files:${NC}"
        echo "================="
        echo "$artifacts_result" | head -50
        echo ""
        
        # Count files
        local file_count
        file_count=$(echo "$artifacts_result" | wc -l)
        log_info "Found approximately $file_count files/directories"
    else
        log_warning "Could not list artifacts. Proceeding with standard download patterns..."
    fi
}

# Download artifacts using gradient CLI
download_with_gradient() {
    local job_id="$1"
    local dest_dir="$2"
    local pattern="$3"
    
    log_download "Downloading $pattern files..."
    
    # Use gradient artifacts download
    if [[ -n "$pattern" ]]; then
        gradient jobs artifacts download --id "$job_id" --dest "$dest_dir" 2>/dev/null || {
            log_warning "Standard download failed, trying alternative method..."
            return 1
        }
    else
        gradient jobs artifacts download --id "$job_id" --dest "$dest_dir" 2>/dev/null || {
            log_warning "Standard download failed, trying alternative method..."
            return 1
        }
    fi
    
    return 0
}

# Organize downloaded files
organize_downloads() {
    local download_dir="$1"
    
    log_info "Organizing downloaded files..."
    
    # Move files to appropriate subdirectories
    if [[ -d "$download_dir" ]]; then
        # Find and move checkpoints
        find "$download_dir" -name "*.ckpt" -o -name "*.pth" -o -name "*.pt" | while read -r file; do
            if [[ -f "$file" ]]; then
                mv "$file" "$download_dir/checkpoints/" 2>/dev/null || true
            fi
        done
        
        # Find and move logs
        find "$download_dir" -name "*.log" -o -name "*metrics*" -o -name "*.json" | while read -r file; do
            if [[ -f "$file" && "$file" != *"metadata.json" ]]; then
                mv "$file" "$download_dir/logs/" 2>/dev/null || true
            fi
        done
        
        # Find and move configs
        find "$download_dir" -name "*.yaml" -o -name "*.yml" -o -name "config*" | while read -r file; do
            if [[ -f "$file" ]]; then
                mv "$file" "$download_dir/configs/" 2>/dev/null || true
            fi
        done
        
        # Find and move wandb files
        find "$download_dir" -path "*wandb*" -type f | while read -r file; do
            if [[ -f "$file" ]]; then
                rel_path=$(echo "$file" | sed "s|$download_dir/||")
                mkdir -p "$download_dir/wandb_logs/$(dirname "$rel_path")"
                mv "$file" "$download_dir/wandb_logs/$rel_path" 2>/dev/null || true
            fi
        done
        
        # Find and move output files (images, plots, etc.)
        find "$download_dir" -name "*.png" -o -name "*.jpg" -o -name "*.pdf" -o -name "*.svg" | while read -r file; do
            if [[ -f "$file" ]]; then
                mv "$file" "$download_dir/outputs/" 2>/dev/null || true
            fi
        done
    fi
    
    log_success "Files organized"
}

# Create metadata file
create_metadata() {
    local job_id="$1"
    local download_dir="$2"
    
    local metadata_file="$download_dir/metadata.json"
    
    log_info "Creating metadata file..."
    
    # Get job information
    local job_info
    job_info=$(get_job_info "$job_id")
    
    # Create metadata
    cat > "$metadata_file" << EOF
{
    "job_id": "$job_id",
    "download_date": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "download_script": "$(basename "$0")",
    "project_root": "$PROJECT_ROOT",
    "download_directory": "$download_dir",
    "job_info": $job_info
}
EOF
    
    log_success "Metadata created: $metadata_file"
}

# Verify downloads
verify_downloads() {
    local download_dir="$1"
    
    if [[ "$VERIFY_DOWNLOADS" == false ]]; then
        return 0
    fi
    
    log_info "Verifying downloads..."
    
    local total_files=0
    local checkpoint_files=0
    local log_files=0
    local config_files=0
    
    # Count files in each directory
    if [[ -d "$download_dir/checkpoints" ]]; then
        checkpoint_files=$(find "$download_dir/checkpoints" -type f | wc -l)
    fi
    
    if [[ -d "$download_dir/logs" ]]; then
        log_files=$(find "$download_dir/logs" -type f | wc -l)
    fi
    
    if [[ -d "$download_dir/configs" ]]; then
        config_files=$(find "$download_dir/configs" -type f | wc -l)
    fi
    
    total_files=$(find "$download_dir" -type f | wc -l)
    
    echo ""
    echo -e "${CYAN}Download Summary:${NC}"
    echo "=================="
    echo "Total files: $total_files"
    echo "Checkpoints: $checkpoint_files"
    echo "Logs: $log_files"
    echo "Configs: $config_files"
    echo "Location: $download_dir"
    echo ""
    
    if [[ $total_files -eq 0 ]]; then
        log_warning "No files were downloaded. This might indicate:"
        log_warning "  - Job has no artifacts"
        log_warning "  - Job is still running"
        log_warning "  - Insufficient permissions"
        log_warning "  - Network issues"
    elif [[ $checkpoint_files -eq 0 && "$LOGS_ONLY" == false ]]; then
        log_warning "No checkpoint files found. Training may not have completed successfully."
    else
        log_success "Download verification completed"
    fi
}

# Main download function
perform_download() {
    local job_id="$1"
    local output_dir="$2"
    
    log_info "Starting download for job: $job_id"
    log_info "Output directory: $output_dir"
    
    # Check if directory exists and handle force flag
    if [[ -d "$output_dir" && "$FORCE_DOWNLOAD" == false ]]; then
        log_warning "Output directory already exists: $output_dir"
        read -p "Continue and merge downloads? (y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Download cancelled by user"
            exit 0
        fi
    fi
    
    # Create directory structure
    create_download_structure "$output_dir"
    
    # Perform the download
    local download_success=false
    
    if download_with_gradient "$job_id" "$output_dir" ""; then
        download_success=true
    else
        log_error "Failed to download artifacts using gradient CLI"
        log_info "This might be due to:"
        log_info "  - Job has no artifacts"
        log_info "  - Network connectivity issues"
        log_info "  - Paperspace API limitations"
    fi
    
    # Organize files
    if [[ "$download_success" == true ]]; then
        organize_downloads "$output_dir"
    fi
    
    # Create metadata
    create_metadata "$job_id" "$output_dir"
    
    # Verify downloads
    verify_downloads "$output_dir"
    
    if [[ "$download_success" == true ]]; then
        log_success "Download completed successfully!"
        log_info "Results saved to: $output_dir"
        
        # Show quick access commands
        echo ""
        echo -e "${BLUE}Quick Access:${NC}"
        echo "View checkpoints: ls -la $output_dir/checkpoints/"
        echo "View logs: ls -la $output_dir/logs/"
        echo "View configs: ls -la $output_dir/configs/"
    else
        log_error "Download completed with errors"
        exit 1
    fi
}

# Main execution
main() {
    check_prerequisites
    parse_arguments "$@"
    
    # Get job ID
    if [[ "$LATEST_JOB" == true ]]; then
        JOB_ID=$(get_latest_job)
        if [[ -z "$JOB_ID" ]]; then
            log_error "No jobs found"
            exit 1
        fi
        log_info "Latest job ID: $JOB_ID"
    fi
    
    # Validate job ID format
    if [[ ! "$JOB_ID" =~ ^js[a-zA-Z0-9]+$ ]]; then
        log_warning "Job ID format looks unusual: $JOB_ID"
        log_warning "Expected format: js123abc456"
    fi
    
    # Check job status
    check_job_status "$JOB_ID"
    
    # Set output directory
    if [[ -z "$OUTPUT_DIR" ]]; then
        OUTPUT_DIR="$PROJECT_ROOT/downloads/job_$JOB_ID"
    fi
    
    # Convert to absolute path
    OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
    
    # List files if requested
    if [[ "$LIST_FILES" == true ]]; then
        list_available_files "$JOB_ID"
        exit 0
    fi
    
    # Perform download
    perform_download "$JOB_ID" "$OUTPUT_DIR"
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi