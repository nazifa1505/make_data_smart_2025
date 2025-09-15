import re
import csv
import uuid

# Party ID mapping
party_id_mapping = {
    179: 'A',    # Arbeiderpartiet
    180: 'FRP',  # Fremskrittspartiet  
    181: 'V',    # Venstre
    182: 'H',    # Høyre
    183: 'KRF',  # Kristelig Folkeparti
    184: 'RØDT', # Rødt
    185: 'SP',   # Senterpartiet
    186: 'MDG',  # Miljøpartiet De Grønne
    187: 'SV'    # Sosialistisk Venstreparti
}

# The additional parties from the CSV header that don't exist in the data
additional_parties = ['DEMN', 'DNI', 'FRED', 'GENP', 'HELSE', 'INP', 'KRISTNE', 'PP', 'PS']

# All parties in the order they should appear in CSV
all_parties = ['A', 'DEMN', 'DNI', 'FRED', 'FRP', 'GENP', 'H', 'HELSE', 'INP', 'KRF', 'KRISTNE', 'MDG', 'PP', 'PS', 'RØDT', 'SP', 'SV', 'V']

def extract_questions_and_positions(file_path):
    """Extract questions and party positions from the HTML file"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all questions with their options and weightings
    questions = []
    
    # Pattern to find questions
    question_pattern = r'id: (\d+),\s+text: "([^"]+)",\s+orderIndex: \d+,\s+category: \{[^}]+\},\s+options: \[((?:[^][]|\[[^]]*\])*)\]'
    
    matches = re.finditer(question_pattern, content, re.DOTALL)
    
    for match in matches:
        question_id = match.group(1)
        question_text = match.group(2)
        options_text = match.group(3)
        
        # Parse options
        party_positions = {}
        
        # Find all options with their weightings
        option_pattern = r'id: \d+,\s+text: "([^"]+)",\s+orderIndex: \d+,\s+weightings: \[((?:[^][]|\[[^]]*\])*)\]'
        option_matches = re.finditer(option_pattern, options_text, re.DOTALL)
        
        for option_match in option_matches:
            option_text = option_match.group(1)
            weightings_text = option_match.group(2)
            
            # Parse weightings
            weighting_pattern = r'actorId: (\d+),\s+weight: ([\d.]+)'
            weighting_matches = re.finditer(weighting_pattern, weightings_text)
            
            for weighting_match in weighting_matches:
                actor_id = int(weighting_match.group(1))
                weight = float(weighting_match.group(2))
                
                if actor_id in party_id_mapping:
                    party_code = party_id_mapping[actor_id]
                    
                    # Convert option text and weight to position value (-2, -1, 0, 1, 2)
                    if "Helt enig" in option_text:
                        position = 2 if weight >= 1.5 else 1
                    elif "Enig" in option_text and "Helt" not in option_text:
                        position = 1
                    elif "Helt uenig" in option_text:
                        position = -2 if weight >= 1.5 else -1
                    elif "Uenig" in option_text and "Helt" not in option_text:
                        position = -1
                    elif "Nøytral" in option_text or "Vet ikke" in option_text:
                        position = 0
                    else:
                        # For other option texts, use weight to determine position
                        if weight >= 1.5:
                            position = 2
                        elif weight >= 1:
                            position = 1
                        else:
                            position = 0
                    
                    party_positions[party_code] = position
        
        # Generate UUID for question
        question_uuid = str(uuid.uuid4())
        
        questions.append({
            'uuid': question_uuid,
            'text': question_text,
            'positions': party_positions
        })
    
    return questions

def write_csv(questions, output_file):
    """Write questions and positions to CSV file"""
    
    # CSV header
    header = ['Spørsmål ID', 'Spørsmål'] + all_parties
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for question in questions:
            row = [question['uuid'], question['text']]
            
            # Add positions for all parties
            for party in all_parties:
                if party in question['positions']:
                    row.append(question['positions'][party])
                elif party in additional_parties:
                    # For parties not in the data, use -2 as default
                    row.append(-2)
                else:
                    # For parties that should be in data but aren't for this question
                    row.append(0)
            
            writer.writerow(row)

if __name__ == "__main__":
    # Extract questions and positions
    questions = extract_questions_and_positions('/Users/punnerud/Downloads/valgomat/tv2/data.html')
    
    print(f"Found {len(questions)} questions")
    
    # Write to CSV
    write_csv(questions, '/Users/punnerud/Downloads/valgomat/tv2/tv2.csv')
    
    print("CSV file created: tv2.csv")
    
    # Print first few questions as sample
    if questions:
        print("\nSample questions:")
        for q in questions[:3]:
            print(f"- {q['text'][:80]}...")
            print(f"  Positions: {q['positions']}")