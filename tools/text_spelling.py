import argparse
from spylls.hunspell import Dictionary

def calculate_spelling_accuracy(file_path):
    # Load the dictionary
    dictionary = Dictionary.from_files('en_US')

    total_words = 0
    correct_words = 0

    # Open the text file
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into words at the underscore
            words = line.strip().split('_')
            for word in words:
                # print("word:", word)
                if word:  # Check if the word is not empty
                    total_words += 1
                    # Check if the word is spelled correctly
                    if dictionary.lookup(word):
                        correct_words += 1

    # Calculate the spelling accuracy
    accuracy = (correct_words / total_words) if total_words > 0 else 0
    return accuracy



def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Calculate spelling accuracy and Type-Token Ratio of a text file.')
    parser.add_argument('file_path', type=str, help='The path to the text file to be analyzed')
    
    # Parse the command line argument
    args = parser.parse_args()

    # Calculate the spelling accuracy
    accuracy = calculate_spelling_accuracy(args.file_path)
    print(f"{accuracy * 100 :.2f}")


if __name__ == '__main__':
    main()

