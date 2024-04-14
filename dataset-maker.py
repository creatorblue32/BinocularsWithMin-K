import csv
import os

def load_progress():
    # Check if a progress file exists
    if os.path.exists('progress.txt'):
        with open('progress.txt', 'r') as file:
            progress = int(file.read().strip())
        return progress
    return 0

def save_progress(progress):
    with open('progress.txt', 'w') as file:
        file.write(str(progress))

def main():
    # List to store the labels and texts
    entries = []

    # Load existing progress
    starting_index = load_progress() + 1

    # Print instructions
    print(f"Continuing from entry {starting_index}.")
    print("Enter up to 40 texts. Type 'done' when finished or after 40 entries.")
    print("The first 20 texts will be labelled as '1' and the next 20 as '0'.")

    # Collecting texts with automated labels
    for i in range(starting_index, 41):
        label = '1' if i <= 20 else '0'
        
        # Get text with possible newlines
        print(f"Entry {i} (Label {label}): (Paste your text and press Enter twice to submit)")
        input_text = []
        while True:
            line = input()
            if line == "":
                break
            input_text.append(line)
        text = "\n".join(input_text)
        if text.lower() == 'done':
            # Save progress and stop the program
            save_progress(i - 1)
            print(f"Progress saved at entry {i - 1}. You can resume later.")
            return
        entries.append((label, text))

    # Writing to CSV
    with open('texts_with_labels.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        writer.writerow(['Label', 'Text'])
        writer.writerows(entries)

    print("Data has been written to 'texts_with_labels.csv'.")
    # Clean up progress file
    if os.path.exists('progress.txt'):
        os.remove('progress.txt')
    print("All entries completed and progress file cleaned.")

if __name__ == "__main__":
    main()
