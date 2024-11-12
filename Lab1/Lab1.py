import re
import nltk
from nltk.tokenize import word_tokenize
import pymorphy3

nltk.download('punkt')

def process_text(file_path):
    """Process text from a file to find noun-adjective pairs with matching gender, number, and case."""
    
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as file:
        text_content = file.read()
    
    # Tokenize the text into individual words
    tokens = word_tokenize(text_content)

    # Initialize the morphological analyzer
    morph_analyzer = pymorphy3.MorphAnalyzer()

    # Variables for tracking results and the previous word's morphological details
    matched_pairs = []
    previous_word = {'word': '', 'POS': '', 'gender': '', 'number': '', 'case': '', 'index': -2}

    # Iterate through the tokens
    for i, token in enumerate(tokens):
        current_word = morph_analyzer.parse(token)[0]  # Get the first (most likely) morphological parsing of the word
        current_tag = current_word.tag
        
        # Check if the word is either a noun or an adjective
        if current_tag.POS in ['NOUN', 'ADJF']:
            # Ensure agreement in gender, number, and case between consecutive words (noun-adjective pair)
            if (i - previous_word['index'] == 1 and
                previous_word['gender'] == current_tag.gender and
                previous_word['number'] == current_tag.number and
                previous_word['case'] == current_tag.case and
                previous_word['POS'] != current_tag.POS):
                
                # Add the matching noun-adjective pair to the results
                matched_pairs.append(f"{previous_word['word']} {current_word.normal_form}")
            
            # Update the previous word's details for future comparison
            previous_word = {
                'word': current_word.normal_form,
                'POS': current_tag.POS,
                'gender': current_tag.gender,
                'number': current_tag.number,
                'case': current_tag.case,
                'index': i
            }

    # Remove duplicates and return unique matching pairs
    return list(set(matched_pairs))

def main():
    """Main function to handle the input file and display results."""
    file_path = 'file_text.txt'  # You can modify the file path if needed
    results = process_text(file_path)

    # Output the final results
    if results:
        print('\nFound matching noun-adjective pairs:')
        for pair in results:
            print(pair)
    else:
        print('No matching pairs found.')

if __name__ == "__main__":
    main()
