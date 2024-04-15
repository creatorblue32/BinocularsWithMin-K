import csv
import os

def save_progress(progress):
    with open('progress.txt', 'w') as file:
        file.write(str(progress))

def main():
    entries = []

    print("Enter text name and text (one entry per line). Type 'done' to finish each entry.")
    while True:
        print("Enter Text Name:")
        text_name = input()
        if text_name.lower() == "done":
            break
        print("Now Text (Enter text and press Enter, then press Enter again when done):")
        input_text = []
        while True:
            line = input()
            if line == "":
                break
            input_text.append(line)
        text = "\n".join(input_text)
        if text.lower() == 'done':
            break
        entries.append((text_name, text))

    with open('texts_with_labels.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        writer.writerow(['Text Name', 'Text'])
        writer.writerows(entries)

    print("Data has been written to 'texts_with_labels.csv'.")

if __name__ == "__main__":
    main()
