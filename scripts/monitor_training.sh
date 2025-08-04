#!/bin/bash
#
# Paperspace Training Monitoring Script
#
# This script provides real-time monitoring of Paperspace GPU training jobs.
# It displays job status, logs, and basic metrics in a user-friendly format.
#
# Usage:
#     ./scripts/monitor_training.sh <job-id>                  # Monitor specific job
#     ./scripts/monitor_training.sh --list                    # List all jobs
#     ./scripts/monitor_training.sh --latest                  # Monitor latest job
#     ./scripts/monitor_training.sh <job-id> --follow         # Follow logs in real-time
#     ./scripts/monitor_training.sh --help                    # Show help
#
# Examples:
#     # Monitor specific job
#     ./scripts/monitor_training.sh js123abc456
#
#     # List all recent jobs
#     ./scripts/monitor_training.sh --list
#
#     # Follow logs of latest job
#     ./scripts/monitor_training.sh --latest --follow
#
#     # Check job status only
#     ./scripts/monitor_training.sh js123abc456 --status-only
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

# Default values
JOB_ID=""
FOLLOW_LOGS=false
STATUS_ONLY=false
LIST_JOBS=false
LATEST_JOB=false
REFRESH_INTERVAL=10

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

log_status() {
    echo -e "${PURPLE}[STATUS]${NC} $1"
}

# Show usage information
show_help() {
    cat << EOF
Paperspace Training Monitoring Script

USAGE:
    $0 <job-id> [OPTIONS]
    $0 --list [OPTIONS]
    $0 --latest [OPTIONS]

ARGUMENTS:
    job-id         Paperspace job ID to monitor (e.g., js123abc456)

OPTIONS:
    --list         List all recent jobs
    --latest       Monitor the most recent job
    --follow       Follow logs in real-time (like tail -f)
    --status-only  Show job status only, no logs
    --interval N   Refresh interval in seconds (default: 10)
    --help         Show this help message

EXAMPLES:
    # Monitor specific job with real-time logs
    $0 js123abc456 --follow

    # Check status of latest job
    $0 --latest --status-only

    # List all jobs from today
    $0 --list

    # Monitor with custom refresh rate
    $0 js123abc456 --interval 5

PREREQUISITES:
    - Paperspace CLI installed and authenticated
    - Active Paperspace account with running jobs

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
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --list)
                LIST_JOBS=true
                shift
                ;;
            --latest)
                LATEST_JOB=true
                shift
                ;;
            --follow)
                FOLLOW_LOGS=true
                shift
                ;;
            --status-only)
                STATUS_ONLY=true
                shift
                ;;
            --interval)
                REFRESH_INTERVAL="$2"
                shift 2
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
    if [[ "$LIST_JOBS" == false && "$LATEST_JOB" == false && -z "$JOB_ID" ]]; then
        log_error "Must specify job ID, --list, or --latest"
        show_help
        exit 1
    fi
    
    if [[ "$LIST_JOBS" == true && -n "$JOB_ID" ]]; then
        log_error "Cannot specify both job ID and --list"
        exit 1
    fi
}

# Format job status with colors
format_status() {
    local status="$1"
    case $status in
        "Pending"|"Queued")
            echo -e "${YELLOW}$status${NC}"
            ;;
        "Running")
            echo -e "${BLUE}$status${NC}"
            ;;
        "Succeeded"|"Completed")
            echo -e "${GREEN}$status${NC}"
            ;;
        "Failed"|"Error"|"Cancelled")
            echo -e "${RED}$status${NC}"
            ;;
        *)
            echo -e "${CYAN}$status${NC}"
            ;;
    esac
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

# List all jobs
list_jobs() {
    log_info "Fetching job list..."
    
    local jobs_json
    jobs_json=$(gradient jobs list --json 2>/dev/null)
    
    if [[ $? -ne 0 ]]; then
        log_error "Failed to fetch job list"
        return 1
    fi
    
    echo -e "\n${CYAN}Recent Paperspace Jobs:${NC}"
    echo "========================"
    
    # Parse JSON and display jobs (requires jq if available)
    if command -v jq &> /dev/null; then
        echo "$jobs_json" | jq -r '.[] | "\(.id) | \(.state) | \(.name) | \(.dtCreated)"' | head -20 | while IFS='|' read -r id state name created; do
            id=$(echo "$id" | xargs)
            state=$(echo "$state" | xargs)
            name=$(echo "$name" | xargs)
            created=$(echo "$created" | xargs)
            
            status_colored=$(format_status "$state")
            echo -e "${id} | ${status_colored} | ${name} | ${created}"
        done
    else
        # Fallback without jq
        echo "$jobs_json" | grep -o '"id":"[^"]*"' | sed 's/"id":"//' | sed 's/"//' | head -10 | while read -r id; do
            echo "Job ID: $id"
        done
        log_warning "Install 'jq' for better job list formatting"
    fi
    
    echo ""
    log_info "Use './scripts/monitor_training.sh <job-id>' to monitor a specific job"
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

# Display job status
show_job_status() {
    local job_id="$1"
    
    local job_info
    job_info=$(get_job_info "$job_id")
    
    if [[ $? -ne 0 ]]; then
        return 1
    fi
    
    # Parse job information
    if command -v jq &> /dev/null; then
        local name state machine_type created started
        name=$(echo "$job_info" | jq -r '.name // "Unknown"')
        state=$(echo "$job_info" | jq -r '.state // "Unknown"')
        machine_type=$(echo "$job_info" | jq -r '.machineType // "Unknown"')
        created=$(echo "$job_info" | jq -r '.dtCreated // "Unknown"')
        started=$(echo "$job_info" | jq -r '.dtStarted // "Not started"')
        
        echo -e "\n${CYAN}Job Status for $job_id:${NC}"
        echo "================================"
        echo "Name: $name"
        echo -e "Status: $(format_status "$state")"
        echo "Machine Type: $machine_type"
        echo "Created: $created"
        echo "Started: $started"
        echo "================================"
        
        # Show additional details for running jobs
        if [[ "$state" == "Running" ]]; then
            log_info "Job is currently running on $machine_type"
        elif [[ "$state" == "Succeeded" ]]; then
            log_success "Job completed successfully"
        elif [[ "$state" == "Failed" ]]; then
            log_error "Job failed. Check logs for details."
        fi
    else
        echo "Job ID: $job_id"
        log_warning "Install 'jq' for detailed job status"
    fi
}

# Show job logs
show_job_logs() {
    local job_id="$1"
    local follow="$2"
    
    log_info "Fetching logs for job: $job_id"
    
    if [[ "$follow" == true ]]; then
        log_info "Following logs (Ctrl+C to stop)..."
        gradient jobs logs --id "$job_id" --follow
    else
        # Show last 50 lines
        gradient jobs logs --id "$job_id" --line 50
    fi
}

# Monitor job continuously
monitor_job() {
    local job_id="$1"
    
    log_info "Starting continuous monitoring for job: $job_id"
    log_info "Press Ctrl+C to stop monitoring"
    
    while true; do
        clear
        echo -e "${PURPLE}=== Paperspace Job Monitor ===${NC}"
        echo "Job ID: $job_id"
        echo "Refresh: every ${REFRESH_INTERVAL}s"
        echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
        
        show_job_status "$job_id"
        
        # Get current status
        local job_info
        job_info=$(get_job_info "$job_id")
        
        if command -v jq &> /dev/null; then
            local state
            state=$(echo "$job_info" | jq -r '.state // "Unknown"')
            
            # Stop monitoring if job is finished
            if [[ "$state" == "Succeeded" || "$state" == "Failed" || "$state" == "Cancelled" ]]; then
                log_info "Job finished with status: $(format_status "$state")"
                break
            fi
        fi
        
        echo ""
        log_info "Next update in ${REFRESH_INTERVAL}s... (Ctrl+C to stop)"
        sleep "$REFRESH_INTERVAL"
    done
}

# Main execution
main() {
    check_prerequisites
    parse_arguments "$@"
    
    if [[ "$LIST_JOBS" == true ]]; then
        list_jobs
        return 0
    fi
    
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
    
    # Show status
    show_job_status "$JOB_ID"
    
    if [[ "$STATUS_ONLY" == true ]]; then
        return 0
    fi
    
    echo ""
    
    # Show logs or monitor
    if [[ "$FOLLOW_LOGS" == true ]]; then
        show_job_logs "$JOB_ID" true
    else
        # Show recent logs
        log_info "Recent logs (last 50 lines):"
        echo "==============================="
        show_job_logs "$JOB_ID" false
        
        echo ""
        log_info "Use --follow to stream logs in real-time"
        log_info "Use --interval N to change refresh rate for monitoring"
    fi
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}Monitoring stopped by user${NC}"; exit 0' INT

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi