import json
import re
from collections import Counter

def analyze_sql_word_positions(file_path, max_position=5):
    """
    Processes a JSON file containing SQL queries and finds the top 5 most occurring
    words along with their probabilities for each word position.
    
    :param file_path: Path to the input JSON file.
    :param max_position: Number of word positions to analyze (e.g., 5 means positions 1 to 5).
    :return: A dictionary mapping each position to a list of tuples (word, count, probability).
    """
    # Load the JSON data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract the SQL queries from the records
    queries = [item['SQL'] for item in data if 'SQL' in item]
    total_queries = len(queries)
    
    if total_queries == 0:
        print("No SQL queries found in the file.")
        return {}
    
    # Tokenize each query into words (converting to uppercase for consistency)
    # Using \b\w+\b helps capture words/identifiers and excludes operators or punctuation
    tokenized_queries = [re.findall(r'\b\w+\b', q.upper()) for q in queries]
    
    results = {}
    
    for pos in range(max_position):
        # Extract the word at the current position for queries that have enough words
        words_at_pos = [tokens[pos] for tokens in tokenized_queries if len(tokens) > pos]
        total_words_at_pos = len(words_at_pos)
        
        # Calculate frequencies and find the top 5
        word_counts = Counter(words_at_pos)
        top_5 = word_counts.most_common(5)
        
        # Structure the results with probabilities
        position_results = []
        for word, count in top_5:
            probability = count / total_words_at_pos if total_words_at_pos > 0 else 0.0
            position_results.append((word, count, probability))
            
        results[pos + 1] = position_results
        
    return results

if __name__ == "__main__":
    json_path = '../Data/BIRD_SQL/minidev/MINIDEV/mini_dev_mysql.json'
    results = analyze_sql_word_positions(json_path)
    print(results)