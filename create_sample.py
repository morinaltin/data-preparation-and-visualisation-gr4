#!/usr/bin/env python3
"""
Create a sample dataset from the large household power consumption files.
Extracts a representative subset for development/testing without needing the full dataset.
"""
import argparse
import os
import sys


def create_sample(input_file: str, output_file: str, num_lines: int = 10000, skip: int = 1):
    """
    Create a sample dataset by taking every Nth line.
    
    Args:
        input_file: Path to the large input file
        output_file: Path to the output sample file
        num_lines: Approximate number of lines to include (excluding header)
        skip: Take every Nth line (1 = consecutive, 10 = every 10th line)
    """
    if not os.path.isfile(input_file):
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        # Always write the header
        header = fin.readline()
        fout.write(header)
        
        line_count = 0
        lines_written = 0
        
        for line in fin:
            line_count += 1
            
            # Take every Nth line
            if line_count % skip == 0:
                fout.write(line)
                lines_written += 1
                
                if lines_written >= num_lines:
                    break
    
    print(f"Created sample: {output_file}")
    print(f"  Lines written: {lines_written + 1} (including header)")
    print(f"  Sampling rate: 1 in {skip}")


def main():
    parser = argparse.ArgumentParser(
        description='Create a sample dataset from large power consumption files.'
    )
    parser.add_argument(
        '-i', '--input',
        default='household_power_consumption.txt',
        help='Input file (default: household_power_consumption.txt)'
    )
    parser.add_argument(
        '-o', '--output',
        default='household_power_consumption_sample.txt',
        help='Output sample file (default: household_power_consumption_sample.txt)'
    )
    parser.add_argument(
        '-n', '--num-lines',
        type=int,
        default=10000,
        help='Approximate number of data lines to include (default: 10000)'
    )
    parser.add_argument(
        '-s', '--skip',
        type=int,
        default=100,
        help='Take every Nth line (default: 100, for ~2M lines -> 20k sample)'
    )
    
    args = parser.parse_args()
    
    create_sample(args.input, args.output, args.num_lines, args.skip)


if __name__ == '__main__':
    main()
