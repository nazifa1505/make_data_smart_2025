#!/usr/bin/env python3
import json
import csv
import re

# Read the data.json file
with open('data.json', 'r', encoding='utf-8') as f:
    content = f.read()

# Extract the JSON part (the file contains JavaScript code)
# Find the JSON.parse() calls and extract the JSON strings
json_match = re.search(r'JSON\.parse\(`(\[.*?\])`\)', content, re.DOTALL)
if json_match:
    json_str = json_match.group(1)
    questions = json.loads(json_str)
else:
    print("Could not find JSON data in file")
    exit(1)

# Prepare CSV data
csv_data = []

# Header row with party names
party_names = set()
for question in questions:
    if 'standpunkter' in question:
        for party_stance in question['standpunkter']:
            party_names.add(party_stance[0])

# Sort party names for consistent column order
party_names = sorted(list(party_names))

# Create header
header = ['Spørsmål ID', 'Spørsmål'] + party_names
csv_data.append(header)

# Process each question
for question in questions:
    if 'tekst' not in question or 'standpunkter' not in question:
        continue
    
    row = [question.get('id', ''), question['tekst']]
    
    # Create a dict of party positions for this question
    party_positions = {}
    for party_stance in question['standpunkter']:
        party_name = party_stance[0]
        points = party_stance[1]
        party_positions[party_name] = points
    
    # Add party scores in the same order as header
    for party in party_names:
        row.append(party_positions.get(party, ''))
    
    csv_data.append(row)

# Write to CSV
with open('data.csv', 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(csv_data)

print(f"Created data.csv with {len(csv_data)-1} questions and {len(party_names)} parties")
print(f"Parties: {', '.join(party_names)}")