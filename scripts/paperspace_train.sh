#!/bin/bash
#
# Paperspace Training Orchestration Script
#
# This script provides a simple console interface for training models on Paperspace GPU.
# It wraps the Python orchestration script with convenient defaults and validation.
#
# Usage:
#     ./scripts/paperspace_train.sh baseline                  # Train baseline VAE
#     ./scripts/paperspace_train.sh conditional --test        # Test conditional VAE
#     ./scripts/paperspace_train.sh --config custom.yaml     # Use custom config
#     ./scripts/paperspace_train.sh --help                   # Show help
#
# Examples:
#     # Quick test run (2 epochs)
#     ./scripts/paperspace_train.sh baseline --test
#     
#     # Full training with specific GPU
#     ./scripts/paperspace_train.sh conditional --instance P5000
#     
#     # Monitor existing job
#     ./scripts/monitor_training.sh <job-id>
#
# Dependencies:
#     - Paperspace CLI installed and authenticated
#     - Python environment with requirements installed
#     - Valid configuration files in configs/
#

set -e  # Exit on any error

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
CONFIG_NAME=""
CONFIG_PATH=""
INSTANCE_TYPE="P4000"
TEST_RUN=false
DRY_RUN=false
VERBOSE=false
PROJECT_ID=""
WANDB_KEY=""

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
Paperspace Training Orchestration Script

USAGE:
    $0 [CONFIG_NAME|--config CONFIG_PATH] [OPTIONS]

PREDEFINED CONFIGS:
    baseline       Use configs/baseline_vae.yaml
    conditional    Use configs/conditional_vae.yaml (Sprint 2+)
    pointcloud     Use configs/pointcloud_vae.yaml (Sprint 3+)
    hierarchical   Use configs/hierarchical_vae.yaml (Sprint 3+)

OPTIONS:
    --config PATH      Use custom configuration file
    --instance TYPE    GPU instance type (P4000, P5000, P6000, V100, P100)
    --project-id ID    Paperspace project ID (auto-detect if not provided)
    --wandb-key KEY    Weights & Biases API key (or set WANDB_API_KEY)
    --test             Run quick test with 2 epochs
    --dry-run          Prepare but don't submit job
    --verbose          Enable verbose logging
    --help             Show this help message

EXAMPLES:
    # Quick test of baseline model
    $0 baseline --test

    # Full training with specific GPU
    $0 conditional --instance P5000

    # Custom config with dry run
    $0 --config my_config.yaml --dry-run

    # Production training with monitoring
    $0 conditional --instance V100 --wandb-key \$WANDB_API_KEY

ENVIRONMENT VARIABLES:
    WANDB_API_KEY      Weights & Biases API key
    PAPERSPACE_API_KEY Paperspace API key
    
PREREQUISITES:
    - Paperspace CLI: pip install gradient
    - Authentication: gradient apiKey set <your-key>
    - W&B account: wandb login (optional)

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
    if [[ ! -f "$PROJECT_ROOT/scripts/train_paperspace.py" ]]; then
        log_error "Python orchestration script not found: scripts/train_paperspace.py"
        exit 1
    fi
    
    # Check Paperspace CLI
    if ! command -v gradient &> /dev/null; then
        log_error "Paperspace CLI not found. Install with: pip install gradient"
        exit 1
    fi
    
    # Check authentication
    if ! gradient apiKey show &> /dev/null; then
        log_error "Paperspace CLI not authenticated. Run: gradient apiKey set <your-key>"
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
            --instance)
                INSTANCE_TYPE="$2"
                shift 2
                ;;
            --project-id)
                PROJECT_ID="$2"
                shift 2
                ;;
            --wandb-key)
                WANDB_KEY="$2"
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

# Validate instance type
validate_instance_type() {
    case $INSTANCE_TYPE in
        P4000|P5000|P6000|V100|P100)
            log_info "Using instance type: $INSTANCE_TYPE"
            ;;
        *)
            log_error "Invalid instance type: $INSTANCE_TYPE"
            log_error "Valid types: P4000, P5000, P6000, V100, P100"
            exit 1
            ;;
    esac
}

# Show configuration summary
show_config_summary() {
    log_info "=== Training Configuration ==="
    echo "Config file: $CONFIG_PATH"
    echo "Instance type: $INSTANCE_TYPE"
    echo "Test run: $TEST_RUN"
    echo "Dry run: $DRY_RUN"
    echo "Verbose: $VERBOSE"
    
    if [[ -n "$PROJECT_ID" ]]; then
        echo "Project ID: $PROJECT_ID"
    else
        echo "Project ID: Auto-detect"
    fi
    
    if [[ -n "$WANDB_KEY" ]]; then
        echo "W&B logging: Enabled"
    elif [[ -n "$WANDB_API_KEY" ]]; then
        echo "W&B logging: Enabled (from env)"
    else
        echo "W&B logging: Disabled"
    fi
    
    echo "================================"
}

# Estimate cost
estimate_cost() {
    local duration_hours
    case $INSTANCE_TYPE in
        P4000) cost_per_hour=0.51 ;;
        P5000) cost_per_hour=0.78 ;;
        P6000) cost_per_hour=1.10 ;;
        V100) cost_per_hour=2.30 ;;
        P100) cost_per_hour=1.73 ;;
        *) cost_per_hour=1.00 ;;
    esac
    
    if [[ "$TEST_RUN" == true ]]; then
        duration_hours=0.25  # 15 minutes
    else
        duration_hours=2.0   # 2 hours typical
    fi
    
    estimated_cost=$(echo "$cost_per_hour * $duration_hours" | bc -l)
    
    log_info "Estimated cost: \$$(printf "%.2f" $estimated_cost) ($cost_per_hour/hour × ${duration_hours}h)"
    
    if [[ "$TEST_RUN" == false ]]; then
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
    local cmd="python scripts/train_paperspace.py"
    cmd="$cmd --config $CONFIG_PATH"
    cmd="$cmd --instance-type $INSTANCE_TYPE"
    
    if [[ -n "$PROJECT_ID" ]]; then
        cmd="$cmd --project-id $PROJECT_ID"
    fi
    
    if [[ -n "$WANDB_KEY" ]]; then
        cmd="$cmd --wandb-key $WANDB_KEY"
    fi
    
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
    log_info "Starting Paperspace training orchestration..."
    
    # Parse arguments
    parse_arguments "$@"
    
    # Check prerequisites
    check_prerequisites
    
    # Validate inputs
    validate_instance_type
    
    # Show configuration
    show_config_summary
    
    # Cost estimation and confirmation
    if command -v bc &> /dev/null; then
        estimate_cost
    else
        log_warning "Install 'bc' for cost estimation"
    fi
    
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
        eval "$python_cmd" 2>&1 | grep -E "(INFO|ERROR|Job|Training)"
    fi
    
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "Training orchestration completed successfully"
        if [[ "$DRY_RUN" == false ]]; then
            log_info "Use './scripts/monitor_training.sh <job-id>' to monitor progress"
            log_info "Use './scripts/download_results.sh <job-id>' to download results"
        fi
    else
        log_error "Training orchestration failed with exit code $exit_code"
        exit $exit_code
    fi
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi