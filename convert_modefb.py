#!/usr/bin/env python3
"""
This script converts modefb.txt to a structured format of mode-value pairs.
Each line in modefb.txt contains 4 pairs of (mode, value), separated by commas.
The script reads each line, parses it into pairs, and writes the result to an output file.
"""

import json
import os
from typing import List, Tuple, Dict

def read_modefb(filename: str) -> List[List[Tuple[int, int]]]:
    """
    Read the modefb.txt file and parse each line into 4 (mode, value) pairs.
    
    Args:
        filename: Path to the modefb.txt file
        
    Returns:
        A list of lines, where each line is a list of 4 (mode, value) tuples
    """
    lines = []
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            # Split the line by commas
            values = line.split(',')
            
            # Ensure we have exactly 8 values (4 pairs)
            if len(values) != 8:
                print(f"Warning: Line {line_num} does not have exactly 8 values: {line}")
                continue
                
            # Parse the values into 4 (mode, value) pairs
            pairs = []
            for i in range(0, 8, 2):
                try:
                    mode = int(values[i])
                    value = int(values[i+1])
                    
                    # Validate mode and value
                    if not (0 <= mode <= 76):
                        print(f"Warning: Invalid mode {mode} on line {line_num}")
                    if value not in (0, 1):
                        print(f"Warning: Invalid value {value} on line {line_num}")
                        
                    pairs.append((mode, value))
                except ValueError:
                    print(f"Warning: Invalid number format on line {line_num}: {values[i]},{values[i+1]}")
                    
            # Add the pairs to the list of lines
            if len(pairs) == 4:
                lines.append(pairs)
                
    return lines

def write_json_output(lines: List[List[Tuple[int, int]]], filename: str) -> None:
    """
    Write the parsed lines to a JSON file.
    
    Args:
        lines: A list of lines, where each line is a list of 4 (mode, value) tuples
        filename: Path to the output JSON file
    """
    # Convert the data to a JSON-serializable format
    json_data = []
    for line_num, line in enumerate(lines, 1):
        line_data = {
            "line_number": line_num,
            "pairs": [{"mode": mode, "value": value} for mode, value in line]
        }
        json_data.append(line_data)
        
    # Write the JSON data to the output file
    with open(filename, 'w') as f:
        json.dump(json_data, f, indent=2)
        
def write_csv_output(lines: List[List[Tuple[int, int]]], filename: str) -> None:
    """
    Write the parsed lines to a CSV file.
    
    Args:
        lines: A list of lines, where each line is a list of 4 (mode, value) tuples
        filename: Path to the output CSV file
    """
    with open(filename, 'w') as f:
        # Write the header
        f.write("line_number,mode1,value1,mode2,value2,mode3,value3,mode4,value4\n")
        
        # Write each line
        for line_num, line in enumerate(lines, 1):
            values = [str(line_num)]
            for mode, value in line:
                values.extend([str(mode), str(value)])
            f.write(",".join(values) + "\n")

def write_python_output(lines: List[List[Tuple[int, int]]], filename: str) -> None:
    """
    Write the parsed lines to a Python file that defines the data as a list of lists.
    
    Args:
        lines: A list of lines, where each line is a list of 4 (mode, value) tuples
        filename: Path to the output Python file
    """
    with open(filename, 'w') as f:
        f.write("# Generated by convert_modefb.py\n")
        f.write("# Each line is a list of 4 (mode, value) tuples\n\n")
        f.write("modefb_data = [\n")
        
        for line in lines:
            f.write("    " + str(line) + ",\n")
            
        f.write("]\n")

def main():
    input_file = "modefb.txt"
    
    # Read the input file
    print(f"Reading {input_file}...")
    lines = read_modefb(input_file)
    print(f"Read {len(lines)} lines")
    
    # Write the output files
    output_dir = "processed"
    os.makedirs(output_dir, exist_ok=True)
    
    json_output = os.path.join(output_dir, "modefb.json")
    csv_output = os.path.join(output_dir, "modefb.csv")
    python_output = os.path.join(output_dir, "modefb_data.py")
    
    print(f"Writing JSON output to {json_output}...")
    write_json_output(lines, json_output)
    
    print(f"Writing CSV output to {csv_output}...")
    write_csv_output(lines, csv_output)
    
    print(f"Writing Python output to {python_output}...")
    write_python_output(lines, python_output)
    
    print("Done!")

if __name__ == "__main__":
    main()

