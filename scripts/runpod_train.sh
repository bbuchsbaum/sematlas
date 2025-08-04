#!/bin/bash
#
# RunPod Training Orchestration Script
#
# This script provides a simple console interface for training models on RunPod GPU.
# It wraps the Python orchestration script with convenient defaults and validation.
#
# Usage:
#     ./scripts/runpod_train.sh baseline                     # Train baseline VAE
#     ./scripts/runpod_train.sh baseline --test              # Test with 2 epochs
#     ./scripts/runpod_train.sh --config custom.yaml         # Use custom config
#     ./scripts/runpod_train.sh --help                      # Show help
#
# Examples:
#     # Quick test run (2 epochs)
#     ./scripts/runpod_train.sh baseline --test
#     
#     # Full training with specific GPU
#     ./scripts/runpod_train.sh baseline --gpu "A100 40GB"
#     
#     # Dry run to see configuration
#     ./scripts/runpod_train.sh baseline --dry-run
#

set -e  # Exit on any error

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
CONFIG_NAME=""
CONFIG_PATH=""
GPU_TYPE="RTX 4090"
TEST_RUN=false
DRY_RUN=false
VERBOSE=false
VOLUME_SIZE=20

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Show usage information
show_help() {
    cat << EOF
RunPod Training Orchestration Script

USAGE:
    $0 [CONFIG_NAME|--config CONFIG_PATH] [OPTIONS]

PREDEFINED CONFIGS:
    baseline       Use configs/baseline_vae.yaml
    conditional    Use configs/conditional_vae.yaml (Sprint 2+)
    pointcloud     Use configs/pointcloud_vae.yaml (Sprint 3+)
    hierarchical   Use configs/hierarchical_vae.yaml (Sprint 3+)

OPTIONS:
    --config PATH      Use custom configuration file
    --gpu TYPE         GPU type (default: "RTX 4090")
                      Options: "RTX 4090", "A100 40GB", "A100 80GB", "H100"
    --test             Run quick test with 2 epochs
    --dry-run          Prepare but don't submit job
    --volume SIZE      Volume size in GB (default: 20)
    --verbose          Enable verbose logging
    --help             Show this help message

GPU OPTIONS:
    "RTX 4090"     - \$0.34/hr - Best price/performance
    "RTX A5000"    - \$0.82/hr - Good VRAM (24GB)
    "RTX A6000"    - \$1.89/hr - High VRAM (48GB)
    "A100 40GB"    - \$1.29/hr - Fast training
    "A100 80GB"    - \$1.89/hr - Very high VRAM
    "H100"         - \$3.58/hr - Fastest available

EXAMPLES:
    # Quick test of baseline model
    $0 baseline --test

    # Full training with A100
    $0 baseline --gpu "A100 40GB"

    # Custom config with dry run
    $0 --config my_config.yaml --dry-run

    # Production training
    $0 baseline --gpu "RTX 4090" --volume 50

ENVIRONMENT VARIABLES:
    RUNPOD_API_KEY     RunPod API key (required)
    
PREREQUISITES:
    - RunPod account with credits
    - Python environment with runpod package
    - Valid API key in .env or environment

EOF
}

# Validate prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if we're in the project root
    if [[ ! -f "$PROJECT_ROOT/train.py" ]]; then
        log_error "Must be run from project root or scripts/ directory"
        log_error "Current path: $PROJECT_ROOT"
        exit 1
    fi
    
    # Check Python orchestration script
    if [[ ! -f "$PROJECT_ROOT/scripts/train_runpod.py" ]]; then
        log_error "Python orchestration script not found: scripts/train_runpod.py"
        exit 1
    fi
    
    # Check for .env file and load it
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        source "$PROJECT_ROOT/.env"
        log_info "Loaded environment from .env"
    fi
    
    # Check RunPod API key
    if [[ -z "$RUNPOD_API_KEY" ]]; then
        log_error "RUNPOD_API_KEY not set. Please add it to .env or export it"
        exit 1
    fi
    
    # Check Python runpod package
    if ! python -c "import runpod" &> /dev/null; then
        log_error "RunPod Python package not found. Install with: pip install runpod"
        exit 1
    fi
    
    log_success "All prerequisites satisfied"
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            baseline|conditional|pointcloud|hierarchical)
                if [[ -n "$CONFIG_NAME" ]]; then
                    log_error "Cannot specify multiple configs"
                    exit 1
                fi
                CONFIG_NAME="$1"
                CONFIG_PATH="configs/${1}_vae.yaml"
                shift
                ;;
            --config)
                if [[ -n "$CONFIG_NAME" ]]; then
                    log_error "Cannot specify both predefined and custom config"
                    exit 1
                fi
                CONFIG_PATH="$2"
                shift 2
                ;;
            --gpu)
                GPU_TYPE="$2"
                shift 2
                ;;
            --test)
                TEST_RUN=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --volume)
                VOLUME_SIZE="$2"
                shift 2
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Validate required arguments
    if [[ -z "$CONFIG_PATH" ]]; then
        log_error "Must specify a configuration. Use predefined name or --config PATH"
        show_help
        exit 1
    fi
    
    # Check if config file exists
    if [[ ! -f "$PROJECT_ROOT/$CONFIG_PATH" ]]; then
        log_error "Configuration file not found: $CONFIG_PATH"
        log_error "Available configs:"
        ls -la "$PROJECT_ROOT/configs/"*.yaml 2>/dev/null || log_error "No config files found in configs/"
        exit 1
    fi
}

# Show configuration summary
show_config_summary() {
    log_info "=== Training Configuration ==="
    echo "Config file: $CONFIG_PATH"
    echo "GPU type: $GPU_TYPE"
    echo "Volume size: ${VOLUME_SIZE}GB"
    echo "Test run: $TEST_RUN"
    echo "Dry run: $DRY_RUN"
    echo "================================"
}

# Estimate cost
estimate_cost() {
    local cost_per_hour
    case "$GPU_TYPE" in
        "RTX 4090") cost_per_hour=0.34 ;;
        "RTX A4000") cost_per_hour=0.56 ;;
        "RTX A5000") cost_per_hour=0.82 ;;
        "RTX A6000") cost_per_hour=1.89 ;;
        "A40") cost_per_hour=0.76 ;;
        "A100 40GB") cost_per_hour=1.29 ;;
        "A100 80GB") cost_per_hour=1.89 ;;
        "H100") cost_per_hour=3.58 ;;
        "V100") cost_per_hour=0.79 ;;
        *) cost_per_hour=1.00 ;;
    esac
    
    local duration_hours
    if [[ "$TEST_RUN" == true ]]; then
        duration_hours=0.25  # 15 minutes
    else
        duration_hours=2.0   # 2 hours typical
    fi
    
    local estimated_cost=$(python -c "print(f'{$cost_per_hour * $duration_hours:.2f}')")
    
    log_info "Estimated cost: \$$estimated_cost (\$$cost_per_hour/hour × ${duration_hours}h)"
    
    if [[ "$TEST_RUN" == false ]] && [[ "$DRY_RUN" == false ]]; then
        log_warning "This is a full training run. Continue? (y/N)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log_info "Training cancelled by user"
            exit 0
        fi
    fi
}

# Build Python command
build_python_command() {
    local cmd="python scripts/train_runpod.py"
    cmd="$cmd --config $CONFIG_PATH"
    cmd="$cmd --gpu-type \"$GPU_TYPE\""
    cmd="$cmd --volume-size $VOLUME_SIZE"
    
    if [[ "$TEST_RUN" == true ]]; then
        cmd="$cmd --test-run"
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        cmd="$cmd --no-submit"
    fi
    
    echo "$cmd"
}

# Main execution
main() {
    log_info "Starting RunPod training orchestration..."
    
    # Parse arguments
    parse_arguments "$@"
    
    # Check prerequisites
    check_prerequisites
    
    # Show configuration
    show_config_summary
    
    # Cost estimation
    estimate_cost
    
    # Build and execute Python command
    local python_cmd
    python_cmd=$(build_python_command)
    
    log_info "Executing: $python_cmd"
    
    # Change to project directory
    cd "$PROJECT_ROOT"
    
    # Execute with proper error handling
    if [[ "$VERBOSE" == true ]]; then
        eval "$python_cmd"
    else
        eval "$python_cmd" 2>&1 | grep -E "(INFO|ERROR|Pod|Training|SSH)"
    fi
    
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "RunPod orchestration completed successfully"
        if [[ "$DRY_RUN" == false ]]; then
            log_info "Use 'runpod pods list' to see your pods"
            log_info "SSH into the pod to monitor training progress"
        fi
    else
        log_error "RunPod orchestration failed with exit code $exit_code"
        exit $exit_code
    fi
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi