#!/bin/bash
#
# RunPod Training Monitoring Script
#
# This script provides monitoring of RunPod GPU training pods.
#
# Usage:
#     ./scripts/monitor_runpod.sh                    # List all pods
#     ./scripts/monitor_runpod.sh <pod-id>           # Get pod details
#     ./scripts/monitor_runpod.sh --help             # Show help
#

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
POD_ID=""
ACTION="list"

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

# Show usage
show_help() {
    cat << EOF
RunPod Training Monitoring Script

USAGE:
    $0                     List all pods
    $0 <pod-id>            Show pod details and connection info
    $0 --help              Show this help message

EXAMPLES:
    # List all pods
    $0

    # Get specific pod info
    $0 abc123def456

RUNPOD CLI COMMANDS:
    runpod pod list                    List all pods
    runpod pod get <pod-id>            Get pod details
    runpod pod stop <pod-id>           Stop a pod (keeps data)
    runpod pod terminate <pod-id>      Terminate pod (deletes data)

SSH ACCESS:
    When you get pod details, use the SSH command shown to connect:
    ssh root@<pod-ip> -p <pod-port>

EOF
}

# Check prerequisites
check_prerequisites() {
    # Load .env if exists
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        source "$PROJECT_ROOT/.env"
    fi
    
    # Check API key
    if [[ -z "$RUNPOD_API_KEY" ]]; then
        log_error "RUNPOD_API_KEY not set. Please add it to .env or export it"
        exit 1
    fi
    
    # Check Python runpod package
    if ! python -c "import runpod" &> /dev/null; then
        log_error "RunPod Python package not found. Install with: pip install runpod"
        exit 1
    fi
}

# List all pods
list_pods() {
    log_info "Fetching RunPod pods..."
    
    python3 << EOF
import os
import runpod
from datetime import datetime

runpod.api_key = os.environ.get('RUNPOD_API_KEY')

try:
    pods = runpod.get_pods()
    
    if not pods:
        print("No pods found")
    else:
        print(f"\n{'ID':<15} {'Name':<30} {'Status':<10} {'GPU':<15} {'Cost/hr':<10}")
        print("-" * 90)
        
        for pod in pods:
            pod_id = pod.get('id', 'N/A')[:12]
            name = pod.get('name', 'N/A')[:28]
            status = pod.get('status', 'N/A')
            gpu = pod.get('gpu_type_id', 'N/A')[:13]
            cost = f"\${pod.get('cost_per_hour', 0):.2f}"
            
            print(f"{pod_id:<15} {name:<30} {status:<10} {gpu:<15} {cost:<10}")
            
except Exception as e:
    print(f"Error fetching pods: {e}")
EOF
}

# Get pod details
get_pod_details() {
    local pod_id="$1"
    log_info "Fetching details for pod: $pod_id"
    
    python3 << EOF
import os
import runpod
from datetime import datetime

runpod.api_key = os.environ.get('RUNPOD_API_KEY')

try:
    pod = runpod.get_pod('$pod_id')
    
    if not pod:
        print("Pod not found")
    else:
        print("\n=== Pod Details ===")
        print(f"ID: {pod.get('id', 'N/A')}")
        print(f"Name: {pod.get('name', 'N/A')}")
        print(f"Status: {pod.get('status', 'N/A')}")
        print(f"GPU Type: {pod.get('gpu_type_id', 'N/A')}")
        print(f"Cost/hour: \${pod.get('cost_per_hour', 0):.2f}")
        print(f"Disk Size: {pod.get('disk_size_gb', 0)}GB")
        
        # SSH access
        ssh_ip = pod.get('public_ip', None)
        ssh_port = pod.get('port', 22)
        
        if ssh_ip:
            print(f"\n=== SSH Access ===")
            print(f"SSH Command: ssh root@{ssh_ip} -p {ssh_port}")
            print(f"Default password may be required on first login")
        
        print(f"\n=== Management Commands ===")
        print(f"Stop pod: runpod pod stop {pod.get('id', '')}")
        print(f"Terminate pod: runpod pod terminate {pod.get('id', '')}")
        print(f"Get logs: Check via SSH connection")
        
except Exception as e:
    print(f"Error fetching pod details: {e}")
EOF
}

# Parse arguments
if [[ $# -eq 0 ]]; then
    ACTION="list"
elif [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_help
    exit 0
else
    ACTION="get"
    POD_ID="$1"
fi

# Main execution
check_prerequisites

case "$ACTION" in
    "list")
        list_pods
        ;;
    "get")
        get_pod_details "$POD_ID"
        ;;
    *)
        log_error "Unknown action: $ACTION"
        show_help
        exit 1
        ;;
esac