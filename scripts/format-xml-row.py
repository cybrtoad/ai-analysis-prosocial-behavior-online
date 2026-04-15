#!/usr/bin/env python3
"""
Format a single-line XML row with many attributes into a readable multi-line format.
Usage: python format-xml-row.py <xml-file> [row-number]
"""

import sys
import re
from pathlib import Path

def format_xml_row(row_line):
    """Format a single XML row element to be more readable."""
    # Extract tag name and attributes
    match = re.match(r'\s*<(\w+)\s+(.*?)\s*/>', row_line, re.DOTALL)
    if not match:
        return row_line  # Return as-is if not matching expected format

    tag_name = match.group(1)
    attributes = match.group(2)

    # Split attributes (handling quoted values with spaces)
    attr_pattern = r'(\w+)="([^"]*)"'
    attrs = re.findall(attr_pattern, attributes)

    # Format as multi-line
    result = f"<{tag_name}\n"
    for name, value in attrs:
        # Truncate very long values for readability
        display_value = value if len(value) < 100 else value[:97] + "..."
        result += f"  {name}=\"{display_value}\"\n"
    result += "/>"

    return result

def main():
    if len(sys.argv) < 2:
        print("Usage: python format-xml-row.py <xml-file> [row-number]")
        print("Example: python format-xml-row.py Posts.xml 1")
        sys.exit(1)

    xml_file = Path(sys.argv[1])
    row_number = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    if not xml_file.exists():
        print(f"Error: File {xml_file} not found")
        sys.exit(1)

    # Read the file and find row elements
    with open(xml_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find all <row> lines
    row_lines = [line for line in lines if '<row' in line.lower()]

    if row_number > len(row_lines):
        print(f"Error: Only {len(row_lines)} rows found, requested row {row_number}")
        sys.exit(1)

    # Format the requested row (1-indexed)
    row_line = row_lines[row_number - 1]
    formatted = format_xml_row(row_line)

    print(formatted)
    print(f"\n--- Row {row_number} of {len(row_lines)} ---")

if __name__ == "__main__":
    main()
