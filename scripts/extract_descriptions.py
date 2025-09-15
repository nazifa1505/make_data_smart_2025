import re
import csv
import uuid

def extract_questions_with_descriptions(file_path):
    """Extract questions with categories and descriptions from the HTML file"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    questions = []
    
    # Pattern to find questions with descriptions and categories
    pattern = r'id: (\d+),\s+text: "([^"]+)",\s+orderIndex: (\d+),\s+category: \{([^}]+)\},\s+options: \[((?:[^][]|\[[^]]*\])*)\],\s+enabled: true,\s+description: "([^"]+)"'
    
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        question_id = match.group(1)
        question_text = match.group(2)
        order_index = match.group(3)
        category_text = match.group(4)
        description = match.group(6)
        
        # Parse category information
        category_id_match = re.search(r'id: (\d+)', category_text)
        category_name_match = re.search(r'name: "([^"]+)"', category_text)
        category_slug_match = re.search(r'slug: "([^"]+)"', category_text)
        
        category_id = category_id_match.group(1) if category_id_match else ""
        category_name = category_name_match.group(1) if category_name_match else ""
        category_slug = category_slug_match.group(1) if category_slug_match else ""
        
        # Clean up description (replace \n with actual newlines)
        description = description.replace('\\n\\n', ' ').replace('\\n', ' ')
        
        # Generate UUID for question
        question_uuid = str(uuid.uuid4())
        
        questions.append({
            'uuid': question_uuid,
            'question_id': question_id,
            'text': question_text,
            'order_index': order_index,
            'category_id': category_id,
            'category_name': category_name,
            'category_slug': category_slug,
            'description': description
        })
    
    return questions

def write_descriptions_csv(questions, output_file):
    """Write questions with descriptions to CSV file"""
    
    # CSV header
    header = ['Spørsmål ID', 'Original ID', 'Spørsmål', 'Rekkefølge', 'Kategori ID', 'Kategori', 'Kategori Slug', 'Beskrivelse']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        # Sort by order index
        questions.sort(key=lambda x: int(x['order_index']))
        
        for question in questions:
            row = [
                question['uuid'],
                question['question_id'],
                question['text'],
                question['order_index'],
                question['category_id'],
                question['category_name'],
                question['category_slug'],
                question['description']
            ]
            writer.writerow(row)

if __name__ == "__main__":
    # Extract questions with descriptions
    questions = extract_questions_with_descriptions('/Users/punnerud/Downloads/valgomat/tv2/data.html')
    
    print(f"Found {len(questions)} questions with descriptions")
    
    # Write to CSV
    write_descriptions_csv(questions, '/Users/punnerud/Downloads/valgomat/tv2/tv2_description.csv')
    
    print("Description file created: tv2_description.csv")
    
    # Print sample
    if questions:
        print("\nSample entries:")
        for q in questions[:3]:
            print(f"- {q['text'][:50]}...")
            print(f"  Category: {q['category_name']}")
            print(f"  Description: {q['description'][:100]}...")
            print()