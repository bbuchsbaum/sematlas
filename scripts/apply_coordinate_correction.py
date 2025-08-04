#!/usr/bin/env python3
"""
Validate coordinate space for deduplicated data.

This script validates that Neurosynth coordinates are in expected MNI152 ranges
and creates the validation log required by S1.1.3 success criteria.
Neurosynth has already transformed all coordinates to MNI152 space during database creation.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.coordinate_transform import validate_coordinate_space, validate_neurosynth_preprocessing


def main():
    """Validate coordinate space for existing data."""
    
    print("=== S1.1.3: Coordinate Space Validation ===\n")
    
    # Step 1: Validate Neurosynth preprocessing understanding
    print("1. Validating Neurosynth preprocessing assumptions...")
    validation_result = validate_neurosynth_preprocessing()
    
    if validation_result:
        print("✅ Neurosynth preprocessing validation PASSED")
    else:
        print("❌ Neurosynth preprocessing validation FAILED")
        return False
    
    # Step 2: Load deduplicated data
    input_path = project_root / "data" / "processed" / "deduplicated_data.csv"
    output_path = project_root / "data" / "processed" / "coordinate_validated_data.csv"
    log_path = project_root / "data" / "processed" / "coordinate_validation_log.json"
    
    print(f"\n2. Loading data from: {input_path}")
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return False
    
    data = pd.read_csv(input_path)
    print(f"✅ Loaded {len(data)} coordinate records from {data['study_id'].nunique()} studies")
    
    # Step 3: Show coordinate space distribution
    print(f"\n3. Coordinate space distribution:")
    space_counts = data['space'].value_counts()
    for space, count in space_counts.items():
        print(f"   {space}: {count} coordinates")
    
    # Step 4: Validate coordinate space (no transformations applied)
    print(f"\n4. Validating coordinate space bounds...")
    validated_data = validate_coordinate_space(
        data,
        output_log_path=str(log_path)
    )
    
    # Step 5: Save validated data (identical to input - no changes made)
    print(f"\n5. Saving validated data to: {output_path}")
    validated_data.to_csv(output_path, index=False)
    
    # Step 6: Report results
    print(f"\n6. Results Summary:")
    print(f"   - Input records: {len(data)}")
    print(f"   - Output records: {len(validated_data)}")
    print(f"   - All coordinates already in MNI space: {all(validated_data['space'] == 'MNI')}")
    print(f"   - Validation log created: {log_path.exists()}")
    
    # Step 7: Validate against success criteria
    print(f"\n7. Success Criteria Validation:")
    
    criteria_pass = True
    
    # Criterion 1: Neurosynth preprocessing understanding validated
    if validation_result:
        print("   ✅ Neurosynth preprocessing assumptions validated")
    else:
        print("   ❌ Neurosynth preprocessing validation failed")
        criteria_pass = False
    
    # Criterion 2: Validation log created
    if log_path.exists():
        print(f"   ✅ coordinate_validation_log.json created with coordinate bounds validation")
    else:
        print("   ❌ coordinate_validation_log.json not created")
        criteria_pass = False
    
    # Criterion 3: No coordinate transformations applied (as intended)
    print("   ✅ No coordinate transformations applied - data preserved as-is")
    
    # Criterion 4: All coordinates preserved exactly
    coordinates_unchanged = True
    original_coords = data[['x', 'y', 'z']]
    validated_coords = validated_data[['x', 'y', 'z']]
    
    if len(original_coords) > 0:
        # Check if coordinates are exactly unchanged
        if len(original_coords) == len(validated_coords):
            diff = abs(original_coords.values - validated_coords.values).max()
            if diff < 1e-10:  # Should be exactly zero
                print("   ✅ All coordinates preserved exactly (no transformations)")
            else:
                print(f"   ❌ Coordinates changed unexpectedly (max diff: {diff})")
                coordinates_unchanged = False
        else:
            print("   ❌ Coordinate count mismatch")
            coordinates_unchanged = False
    else:
        print("   ✅ No coordinates to verify")
    
    criteria_pass = criteria_pass and coordinates_unchanged
    
    print(f"\n{'='*50}")
    if criteria_pass:
        print("🎉 S1.1.3 SUCCESS: All criteria PASSED")
        print("Ready to proceed to S1.1.4")
    else:
        print("❌ S1.1.3 FAILED: Some criteria not met")
        print("Review implementation before proceeding")
    
    return criteria_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)