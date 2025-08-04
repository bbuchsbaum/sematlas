#!/bin/bash
#
# Paperspace Training Cost Report Script
#
# This script generates detailed cost reports for Paperspace GPU training usage.
# It tracks job costs, provides spending summaries, and helps with budget monitoring.
#
# Usage:
#     ./scripts/cost_report.sh                            # Generate report for last 30 days
#     ./scripts/cost_report.sh --period 7                 # Report for last 7 days
#     ./scripts/cost_report.sh --month 2025-01            # Report for specific month
#     ./scripts/cost_report.sh --job-id js123abc456       # Cost for specific job
#     ./scripts/cost_report.sh --export-csv report.csv    # Export to CSV
#     ./scripts/cost_report.sh --help                     # Show help
#
# Examples:
#     # Weekly cost summary
#     ./scripts/cost_report.sh --period 7
#
#     # Monthly detailed report
#     ./scripts/cost_report.sh --month 2025-01 --detailed
#
#     # Export all jobs to CSV
#     ./scripts/cost_report.sh --period 90 --export-csv training_costs.csv
#
#     # Budget monitoring
#     ./scripts/cost_report.sh --budget 50 --period 30
#
# Features:
#     - Per-job cost calculation
#     - Instance type utilization analysis
#     - Time-based cost breakdowns
#     - Budget monitoring and alerts
#     - CSV export for external analysis
#     - Cost optimization recommendations
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
PERIOD_DAYS=30
SPECIFIC_MONTH=""
JOB_ID=""
EXPORT_CSV=""
BUDGET_LIMIT=""
DETAILED_REPORT=false
INCLUDE_RUNNING=false
SHOW_RECOMMENDATIONS=true

# Machine type pricing (USD per hour)
declare -A MACHINE_COSTS=(
    ["P4000"]=0.51
    ["P5000"]=0.78
    ["P6000"]=1.10
    ["V100"]=2.30
    ["P100"]=1.73
    ["RTX4000"]=0.56
    ["RTX5000"]=0.82
    ["A4000"]=0.76
    ["A5000"]=1.38
    ["A6000"]=1.89
)

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

log_cost() {
    echo -e "${PURPLE}[COST]${NC} $1"
}

# Show usage information
show_help() {
    cat << EOF
Paperspace Training Cost Report Script

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --period DAYS        Report period in days (default: 30)
    --month YYYY-MM      Report for specific month (e.g., 2025-01)
    --job-id ID          Cost report for specific job
    --export-csv FILE    Export detailed report to CSV
    --budget AMOUNT      Set budget limit for monitoring (USD)
    --detailed           Show detailed per-job breakdown
    --include-running    Include currently running jobs
    --no-recommendations Skip cost optimization recommendations
    --help               Show this help message

EXAMPLES:
    # Weekly cost summary
    $0 --period 7

    # Detailed monthly report
    $0 --month 2025-01 --detailed

    # Budget monitoring with alerts
    $0 --budget 100 --period 30

    # Export to CSV for analysis
    $0 --period 90 --export-csv quarterly_costs.csv

    # Single job cost analysis
    $0 --job-id js123abc456 --detailed

COST BREAKDOWN:
    - Per-job compute costs
    - Storage costs (data transfer)
    - Time-based analysis
    - Instance type utilization
    - Cost optimization opportunities

MACHINE TYPES & COSTS (USD/hour):
    P4000:   \$0.51    RTX4000: \$0.56    A4000:  \$0.76
    P5000:   \$0.78    RTX5000: \$0.82    A5000:  \$1.38
    P6000:   \$1.10    P100:    \$1.73    A6000:  \$1.89
    V100:    \$2.30

PREREQUISITES:
    - Paperspace CLI installed and authenticated
    - 'jq' for JSON processing (recommended)
    - 'bc' for calculations

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
    
    if ! command -v bc &> /dev/null; then
        log_warning "Install 'bc' for accurate cost calculations"
    fi
    
    if ! command -v jq &> /dev/null; then
        log_warning "Install 'jq' for enhanced JSON processing"
    fi
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --period)
                PERIOD_DAYS="$2"
                shift 2
                ;;
            --month)
                SPECIFIC_MONTH="$2"
                shift 2
                ;;
            --job-id)
                JOB_ID="$2"
                shift 2
                ;;
            --export-csv)
                EXPORT_CSV="$2"
                shift 2
                ;;
            --budget)
                BUDGET_LIMIT="$2"
                shift 2
                ;;
            --detailed)
                DETAILED_REPORT=true
                shift
                ;;
            --include-running)
                INCLUDE_RUNNING=true
                shift
                ;;
            --no-recommendations)
                SHOW_RECOMMENDATIONS=false
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
    
    # Validate period
    if [[ ! "$PERIOD_DAYS" =~ ^[0-9]+$ ]]; then
        log_error "Period must be a positive integer"
        exit 1
    fi
    
    # Validate month format
    if [[ -n "$SPECIFIC_MONTH" && ! "$SPECIFIC_MONTH" =~ ^[0-9]{4}-[0-9]{2}$ ]]; then
        log_error "Month must be in YYYY-MM format"
        exit 1
    fi
    
    # Validate budget
    if [[ -n "$BUDGET_LIMIT" && ! "$BUDGET_LIMIT" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        log_error "Budget must be a positive number"
        exit 1
    fi
}

# Get jobs for the specified period
get_jobs_data() {
    local jobs_json
    
    log_info "Fetching job data..."
    
    if [[ -n "$JOB_ID" ]]; then
        # Single job
        jobs_json=$(gradient jobs show --id "$JOB_ID" --json 2>/dev/null)
        if [[ $? -ne 0 ]]; then
            log_error "Failed to fetch job: $JOB_ID"
            exit 1
        fi
        echo "[$jobs_json]"  # Wrap in array for consistency
    else
        # Multiple jobs
        jobs_json=$(gradient jobs list --json 2>/dev/null)
        if [[ $? -ne 0 ]]; then
            log_error "Failed to fetch job list"
            exit 1
        fi
        echo "$jobs_json"
    fi
}

# Calculate job duration in hours
calculate_job_duration() {
    local start_time="$1"
    local end_time="$2"
    
    if [[ "$start_time" == "null" || "$start_time" == "" ]]; then
        echo "0"
        return
    fi
    
    if [[ "$end_time" == "null" || "$end_time" == "" ]]; then
        # Job is running, calculate from start to now
        if [[ "$INCLUDE_RUNNING" == true ]]; then
            end_time=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        else
            echo "0"
            return
        fi
    fi
    
    # Convert to timestamps (macOS compatible)
    if command -v gdate &> /dev/null; then
        # GNU date (from coreutils)
        local start_epoch=$(gdate -d "$start_time" +%s 2>/dev/null || echo "0")
        local end_epoch=$(gdate -d "$end_time" +%s 2>/dev/null || echo "0")
    else
        # macOS date
        local start_epoch=$(date -j -f "%Y-%m-%dT%H:%M:%SZ" "$start_time" +%s 2>/dev/null || echo "0")
        local end_epoch=$(date -j -f "%Y-%m-%dT%H:%M:%SZ" "$end_time" +%s 2>/dev/null || echo "0")
    fi
    
    if [[ $start_epoch -eq 0 || $end_epoch -eq 0 ]]; then
        echo "0"
        return
    fi
    
    local duration_seconds=$((end_epoch - start_epoch))
    local duration_hours=$(echo "scale=3; $duration_seconds / 3600" | bc -l 2>/dev/null || echo "0")
    
    echo "$duration_hours"
}

# Calculate job cost
calculate_job_cost() {
    local machine_type="$1"
    local duration_hours="$2"
    
    local cost_per_hour="${MACHINE_COSTS[$machine_type]:-1.00}"
    local total_cost=$(echo "scale=2; $duration_hours * $cost_per_hour" | bc -l 2>/dev/null || echo "0")
    
    echo "$total_cost"
}

# Filter jobs by date range
filter_jobs_by_date() {
    local jobs_json="$1"
    
    if [[ -n "$SPECIFIC_MONTH" ]]; then
        # Filter by specific month
        if command -v jq &> /dev/null; then
            echo "$jobs_json" | jq --arg month "$SPECIFIC_MONTH" '[.[] | select(.dtCreated | startswith($month))]'
        else
            echo "$jobs_json"
        fi
    elif [[ -n "$JOB_ID" ]]; then
        # Single job, no filtering needed
        echo "$jobs_json"
    else
        # Filter by period days
        local cutoff_date
        if command -v gdate &> /dev/null; then
            cutoff_date=$(gdate -d "$PERIOD_DAYS days ago" -u +"%Y-%m-%dT%H:%M:%SZ")
        else
            cutoff_date=$(date -u -v-${PERIOD_DAYS}d +"%Y-%m-%dT%H:%M:%SZ")
        fi
        
        if command -v jq &> /dev/null; then
            echo "$jobs_json" | jq --arg cutoff "$cutoff_date" '[.[] | select(.dtCreated >= $cutoff)]'
        else
            echo "$jobs_json"
        fi
    fi
}

# Generate cost summary
generate_cost_summary() {
    local jobs_json="$1"
    
    log_info "Generating cost summary..."
    
    local total_cost=0
    local total_hours=0
    local job_count=0
    declare -A machine_usage
    declare -A machine_costs
    declare -A daily_costs
    
    # Process each job
    if command -v jq &> /dev/null; then
        while IFS=$'\t' read -r id name state machine_type start_time end_time created; do
            if [[ -z "$id" ]]; then continue; fi
            
            job_count=$((job_count + 1))
            
            # Calculate duration and cost
            local duration_hours
            duration_hours=$(calculate_job_duration "$start_time" "$end_time")
            
            local job_cost
            job_cost=$(calculate_job_cost "$machine_type" "$duration_hours")
            
            # Update totals
            total_cost=$(echo "$total_cost + $job_cost" | bc -l)
            total_hours=$(echo "$total_hours + $duration_hours" | bc -l)
            
            # Update machine usage
            machine_usage["$machine_type"]=$(echo "${machine_usage[$machine_type]:-0} + $duration_hours" | bc -l)
            machine_costs["$machine_type"]=$(echo "${machine_costs[$machine_type]:-0} + $job_cost" | bc -l)
            
            # Update daily costs (simplified - just use created date)
            local day=$(echo "$created" | cut -d'T' -f1)
            daily_costs["$day"]=$(echo "${daily_costs[$day]:-0} + $job_cost" | bc -l)
            
            # Show detailed job info if requested
            if [[ "$DETAILED_REPORT" == true ]]; then
                printf "%-12s %-20s %-10s %-8s %8.2f %8.2f %s\n" \
                    "$id" "${name:0:20}" "$state" "$machine_type" \
                    "$duration_hours" "$job_cost" "$created"
            fi
            
        done < <(echo "$jobs_json" | jq -r '.[] | [.id, .name, .state, .machineType, .dtStarted, .dtFinished, .dtCreated] | @tsv')
    else
        log_warning "Limited functionality without 'jq'. Install jq for detailed reports."
        job_count=$(echo "$jobs_json" | grep -o '"id"' | wc -l)
        total_cost="Unknown"
    fi
    
    # Display summary
    echo ""
    echo -e "${CYAN}=== COST SUMMARY ===${NC}"
    echo "Period: $(if [[ -n "$SPECIFIC_MONTH" ]]; then echo "$SPECIFIC_MONTH"; elif [[ -n "$JOB_ID" ]]; then echo "Single job"; else echo "Last $PERIOD_DAYS days"; fi)"
    echo "Report Date: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    printf "%-20s %s\n" "Total Jobs:" "$job_count"
    printf "%-20s %.2f hours\n" "Total Runtime:" "$total_hours"
    printf "%-20s \$%.2f\n" "Total Cost:" "$total_cost"
    
    if [[ $(echo "$total_hours > 0" | bc -l) -eq 1 ]]; then
        local avg_cost_per_hour=$(echo "scale=2; $total_cost / $total_hours" | bc -l)
        printf "%-20s \$%.2f/hour\n" "Average Cost:" "$avg_cost_per_hour"
    fi
    
    # Budget check
    if [[ -n "$BUDGET_LIMIT" ]]; then
        echo ""
        local budget_used=$(echo "scale=1; $total_cost / $BUDGET_LIMIT * 100" | bc -l)
        printf "%-20s \$%.2f / \$%.2f (%.1f%%)\n" "Budget Usage:" "$total_cost" "$BUDGET_LIMIT" "$budget_used"
        
        if [[ $(echo "$total_cost > $BUDGET_LIMIT" | bc -l) -eq 1 ]]; then
            log_warning "Budget exceeded by \$$(echo "$total_cost - $BUDGET_LIMIT" | bc -l)"
        elif [[ $(echo "$budget_used > 80" | bc -l) -eq 1 ]]; then
            log_warning "Budget usage above 80%"
        fi
    fi
    
    # Machine type breakdown
    if [[ ${#machine_usage[@]} -gt 0 ]]; then
        echo ""
        echo -e "${CYAN}Machine Type Usage:${NC}"
        echo "===================="
        printf "%-10s %8s %10s %12s\n" "Type" "Hours" "Cost" "Efficiency"
        echo "----------------------------------------"
        
        for machine_type in "${!machine_usage[@]}"; do
            local hours="${machine_usage[$machine_type]}"
            local cost="${machine_costs[$machine_type]}"
            local rate="${MACHINE_COSTS[$machine_type]:-1.00}"
            
            printf "%-10s %8.2f \$%8.2f \$%.2f/hr\n" \
                "$machine_type" "$hours" "$cost" "$rate"
        done
    fi
    
    # Export CSV if requested
    if [[ -n "$EXPORT_CSV" ]]; then
        export_to_csv "$jobs_json" "$EXPORT_CSV"
    fi
}

# Export detailed report to CSV
export_to_csv() {
    local jobs_json="$1"
    local csv_file="$2"
    
    log_info "Exporting to CSV: $csv_file"
    
    # Create CSV header
    echo "job_id,name,state,machine_type,created,started,finished,duration_hours,cost_usd" > "$csv_file"
    
    if command -v jq &> /dev/null; then
        echo "$jobs_json" | jq -r '.[] | [.id, .name, .state, .machineType, .dtCreated, .dtStarted, .dtFinished] | @csv' | while IFS=',' read -r line; do
            # Parse the line and calculate cost
            local id name state machine_type created started finished
            eval "read id name state machine_type created started finished <<< \"$line\""
            
            local duration_hours
            duration_hours=$(calculate_job_duration "$started" "$finished")
            
            local cost
            cost=$(calculate_job_cost "$machine_type" "$duration_hours")
            
            echo "$line,$duration_hours,$cost" >> "$csv_file"
        done
    else
        log_warning "CSV export requires 'jq'. Basic CSV created with limited data."
    fi
    
    log_success "CSV exported: $csv_file"
}

# Show cost optimization recommendations
show_recommendations() {
    if [[ "$SHOW_RECOMMENDATIONS" == false ]]; then
        return
    fi
    
    echo ""
    echo -e "${CYAN}=== COST OPTIMIZATION RECOMMENDATIONS ===${NC}"
    echo ""
    
    # General recommendations
    echo "💡 Cost Optimization Tips:"
    echo "  • Use P4000 for development/testing (\$0.51/hr)"
    echo "  • Reserve V100/A6000 for production training only"
    echo "  • Monitor jobs actively to avoid idle time"
    echo "  • Use --test flag for quick validation runs"
    echo "  • Consider batch processing multiple experiments"
    echo ""
    
    # Budget-specific recommendations
    if [[ -n "$BUDGET_LIMIT" ]]; then
        local weekly_budget=$(echo "scale=2; $BUDGET_LIMIT / 4" | bc -l)
        echo "📊 Budget Management:"
        echo "  • Weekly budget target: \$$(printf "%.2f" "$weekly_budget")"
        echo "  • Track spending with: ./scripts/cost_report.sh --period 7"
        echo "  • Set up cost alerts for budget monitoring"
        echo ""
    fi
    
    # Machine type recommendations
    echo "🖥️  Machine Type Selection:"
    echo "  • P4000: Development, small models, testing"
    echo "  • P5000: Medium models, experimentation"
    echo "  • V100: Large models, production training"
    echo "  • A6000: Latest models, maximum performance"
    echo ""
    
    echo "📈 Monitoring Commands:"
    echo "  • Weekly costs: ./scripts/cost_report.sh --period 7"
    echo "  • Monthly report: ./scripts/cost_report.sh --month \$(date +%Y-%m)"
    echo "  • Job monitoring: ./scripts/monitor_training.sh <job-id>"
}

# Main execution
main() {
    check_prerequisites
    parse_arguments "$@"
    
    # Get jobs data
    local jobs_json
    jobs_json=$(get_jobs_data)
    
    # Filter by date range
    local filtered_jobs
    filtered_jobs=$(filter_jobs_by_date "$jobs_json")
    
    # Show detailed header if requested
    if [[ "$DETAILED_REPORT" == true ]]; then
        echo ""
        echo -e "${CYAN}Detailed Job Report:${NC}"
        echo "===================="
        printf "%-12s %-20s %-10s %-8s %8s %8s %s\n" \
            "Job ID" "Name" "State" "Machine" "Hours" "Cost" "Created"
        echo "--------------------------------------------------------------------------------"
    fi
    
    # Generate cost summary
    generate_cost_summary "$filtered_jobs"
    
    # Show recommendations
    show_recommendations
    
    echo ""
    log_success "Cost report generated successfully"
    
    if [[ -n "$EXPORT_CSV" ]]; then
        log_info "CSV export: $EXPORT_CSV"
    fi
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi