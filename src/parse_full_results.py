import csv

def parse_and_save_to_csv(input_file, output_csv):
    with open(input_file, 'r') as file:
        data = file.readlines()
    
    datasets = []
    current_dataset = {}
    category = None 
    
    for line in data:
        if "In dataset" in line:
            if current_dataset:
                datasets.append(current_dataset)
            current_dataset = {'name': ("WikiMia"+line[10:] if "length" in line else line[10:]).strip(), 'seen': [], 'unseen': []}
        elif "Seen Scores:" in line:
            category = 'seen'
            scores = line.split(":")[1].strip().strip('[]').split(', ')
            current_dataset[category].extend([(float(score), None) for score in scores])
        elif "Unseen Scores:" in line:
            category = 'unseen'
            scores = line.split(":")[1].strip().strip('[]').split(', ')
            current_dataset[category].extend([(float(score), None) for score in scores])
        elif "Seen Minks:" in line:
            minks = line.split(":")[1].strip().strip('[]').split(', ')
            for i, mink in enumerate(minks):
                if current_dataset['seen'][i][1] is None:
                    current_dataset['seen'][i] = (current_dataset['seen'][i][0], float(mink))
        elif "Unseen Minks:" in line:
            minks = line.split(":")[1].strip().strip('[]').split(', ')
            for i, mink in enumerate(minks):
                if current_dataset['unseen'][i][1] is None:
                    current_dataset['unseen'][i] = (current_dataset['unseen'][i][0], float(mink))
    
    if current_dataset:
        datasets.append(current_dataset)

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['DatasetName', 'Seen/Unseen', 'Binoculars Score', 'Mink Score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for dataset in datasets:
            for seen_data in dataset['seen']:
                writer.writerow({'DatasetName': dataset['name'], 'Seen/Unseen': 'Seen', 
                                 'Binoculars Score': seen_data[0], 'Mink Score': seen_data[1] * 1e9})
            for unseen_data in dataset['unseen']:
                writer.writerow({'DatasetName': dataset['name'], 'Seen/Unseen': 'Unseen', 
                                 'Binoculars Score': unseen_data[0], 'Mink Score': unseen_data[1] * 1e9})

parse_and_save_to_csv('full_test.txt', 'data/full_test_data.csv')
