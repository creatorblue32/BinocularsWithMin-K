import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

def clean_data(value):
    if isinstance(value, str):
        left_bracket = value.find('[')
        right_bracket = value.find(']')
        if left_bracket != -1 and right_bracket != -1:
            extracted = value[left_bracket+1:right_bracket]
            try:
                return float(extracted)
            except ValueError:
                print("VALUE ERROR:", value)
                return value
    return float(value) if isinstance(value, str) else value

def load_and_clean_data(filepath):
    try:
        data = pd.read_csv(filepath)
        return data.applymap(clean_data)
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

data_filepath = '../data/bookmia_by_ratio_results.csv'
new_data_filepath = '../data/vulnerable_texts_scores.csv'

data = load_and_clean_data(data_filepath)
data = data.iloc[1:]
X = data.drop(columns=['label', 'binoculars_score'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

new_data = pd.read_csv(new_data_filepath)
new_data = new_data.drop('title', axis=1) 
new_data = new_data.applymap(clean_data)
X_new = scaler.transform(new_data) 
print(new_data)

# Threshold-based classification
thresholds = (X_train[y_train == 0].mean() + X_train[y_train == 1].mean()) / 2
print(thresholds)
y_pred_threshold = (X_test > thresholds).mean(axis=1) > 0.5
print(f'Threshold Model Accuracy: {accuracy_score(y_test, y_pred_threshold)}')

new_data_labels = np.ones(len(new_data))

accuracies = {}

new_data_titles = pd.read_csv(new_data_filepath)

for feature in X.columns:
    threshold = thresholds[feature]
    predictions = (new_data[feature] < threshold).astype(int)
    accuracy = accuracy_score(new_data_labels, predictions)
    accuracies[feature] = accuracy
    print(f'Accuracy for {feature} using threshold: {threshold} is {accuracy}')

print()
print()
print()
print()

for i in range(len(new_data_titles['title'])):
    #print(new_data_titles['title'][i])
    predictions = (new_data['min_k_prob_0.05'][i] < 3.809450257187945e-08).astype(int)
    print(str(new_data['min_k_prob_0.05'][i]) + " is less than threshold " +str(3.809450257187945e-08) + "? " + str((new_data['min_k_prob_0.05'][i] < 3.809450257187945e-08)))

print()
print()
print()
print()

accuracies = {}
for feature in X.columns:
    threshold = thresholds[feature]
    predictions = (X_test[feature] < threshold).astype(int)
    accuracy = accuracy_score(y_test, predictions)
    accuracies[feature] = accuracy
    print(f'Accuracy for {feature} using threshold: {threshold} is {accuracy}')
