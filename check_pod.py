#!/usr/bin/env python3
import os
import runpod

# Set API key
runpod.api_key = os.environ.get('RUNPOD_API_KEY', 'rpa_FOSQ48T9HZANHTK5MZPFRL7XK6DCAJUNZE17K42W1vnr73')

# Get pod
pod = runpod.get_pod('y8rgiwnfy0kpua')

print('Pod Status:')
print(f"Name: {pod['name']}")
print(f"GPU: {pod['machine']['gpuDisplayName']}")
print(f"Cost: ${pod['costPerHr']}/hr")
print(f"Uptime: {pod['uptimeSeconds']} seconds")

runtime = pod.get('runtime', {})
if runtime and runtime.get('ports'):
    ports = runtime['ports'][0]
    print(f"\nSSH Access:")
    print(f"ssh root@{ports['ip']} -p {ports['externalPort']}")
else:
    print("\nSSH access not yet available. Pod may still be initializing.")
    
print(f"\nDocker command: {pod['dockerArgs'][:100]}...")
print("\nTo terminate pod: runpod.terminate_pod('y8rgiwnfy0kpua')")