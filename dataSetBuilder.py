import pandas as pd
import ast

def process_csv(file_path):
    df = pd.read_csv(file_path)
    processed_data = []

    for _, row in df.iterrows():
        question = row['question']
        best_answer = row['best_answer']
        
        # Convert string representations of lists to actual lists
        correct_answers = ast.literal_eval(row['correct_answers'])
        incorrect_answers = ast.literal_eval(row['incorrect_answers'])
        
        # Combine all answers
        all_answers = correct_answers + incorrect_answers
        
        # Create pairs of questions and answers
        for answer in all_answers:
            processed_data.append(f"Question: {question} Answer: {answer}")
        
        # Add the best answer separately
        processed_data.append(f"Question: {question} Best Answer: {best_answer}")

    return processed_data

def save_processed_data(processed_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(f"{item}\n")

if __name__ == "__main__":
    processed_data = process_csv('sampleData.csv')
    save_processed_data(processed_data, 'processed_data.txt')