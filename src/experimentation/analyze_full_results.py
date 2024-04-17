import pandas as pd

# Load data
data = pd.read_csv('../data/full_test_data.csv')

# Thresholds
BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843 
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527
MINK_SEEN_THRESHOLD = 399.3

def calculate_averages():
    seen_data = data[data['Seen/Unseen'] == 'Seen']
    unseen_data = data[data['Seen/Unseen'] == 'Unseen']
    
    seen_binoculars_avg = seen_data['Binoculars Score'].mean()
    seen_mink_avg = seen_data['Mink Score'].mean()
    unseen_binoculars_avg = unseen_data['Binoculars Score'].mean()
    unseen_mink_avg = unseen_data['Mink Score'].mean()
    
    print(f"Average Binoculars Score for Seen Samples: {seen_binoculars_avg}")
    print(f"Average Mink Score for Seen Samples: {seen_mink_avg}")
    print(f"Average Binoculars Score for Unseen Samples: {unseen_binoculars_avg}")
    print(f"Average Mink Score for Unseen Samples: {unseen_mink_avg}")

def threshold_accuracy():
    data['Predicted Seen/Unseen'] = data['Mink Score'].apply(lambda x: 'Unseen' if x > MINK_SEEN_THRESHOLD else 'Seen')

    true_positives = data[(data['Predicted Seen/Unseen'] == 'Seen') & (data['Seen/Unseen'] == 'Seen')].shape[0]
    true_negatives = data[(data['Predicted Seen/Unseen'] == 'Unseen') & (data['Seen/Unseen'] == 'Unseen')].shape[0]
    false_positives = data[(data['Predicted Seen/Unseen'] == 'Seen') & (data['Seen/Unseen'] == 'Unseen')].shape[0]
    false_negatives = data[(data['Predicted Seen/Unseen'] == 'Unseen') & (data['Seen/Unseen'] == 'Seen')].shape[0]

    accuracy = (true_positives + true_negatives) / data.shape[0]
    print(f"Accuracy: {accuracy}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")

def count_false_positives_saved():
    binoculars_false_positives =  data[(data['Binoculars Score'] < BINOCULARS_ACCURACY_THRESHOLD)].shape[0]
    count = data[(data['Binoculars Score'] < BINOCULARS_ACCURACY_THRESHOLD) & (data['Mink Score'] < MINK_SEEN_THRESHOLD)].shape[0]
    low_fpr_count = data[(data['Binoculars Score'] < BINOCULARS_FPR_THRESHOLD) & (data['Mink Score'] < MINK_SEEN_THRESHOLD)].shape[0]
    print(f"Count of samples below both Accuracy Threshold: {count} and below Low_FPR Threshold {low_fpr_count} with total {data['Seen/Unseen'].shape[0]}")
    print(f"Count of samples false positives: {binoculars_false_positives}")

calculate_averages()
threshold_accuracy()
count_false_positives_saved()
