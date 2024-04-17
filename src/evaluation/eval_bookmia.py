import binoculars
from datasets import load_dataset
import pandas as pd

observer_model, performer_model = binoculars.create_models()
binoculars_obj = binoculars.Binoculars(observer_model, performer_model, "low-fpr", 512)


dataset = load_dataset("swj0419/BookMIA",split=f"train")

def process_bookmia_data(dataset):
    results = []
    for i in range(len(dataset['label'])):
        print("Processing:" +str(i))
        binoculars_score, min_k_result = binoculars_obj.compute_score_and_all_min_k(dataset['snippet'][i])
        if i == 1:
          #Just for verification
          print("Computing Scores on Data Like: ")
          print(dataset['snippet'][i])
          print("Labelled:")
          print(dataset['label'][i])

        result = {
            'label': dataset['label'][i],
            'binoculars_score': binoculars_score
        }
        result.update({f'min_k_prob_{ratio}': probs for ratio, probs in min_k_result.items()})
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv('mink_by_ratio_results.csv', index=False)
    print("Saved results to 'results.csv'")

    thresholds = {}
    accuracies = {}
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        avg_label_1 = results_df[results_df['label'] == 1][f'min_k_prob_{ratio}'].mean()
        avg_label_0 = results_df[results_df['label'] == 0][f'min_k_prob_{ratio}'].mean()
        threshold = (avg_label_1 + avg_label_0) / 2
        thresholds[ratio] = threshold

        # Compute accuracy
        correct = results_df.apply(lambda row: 1 if (row[f'min_k_prob_{ratio}'] > threshold) == row['label'] else 0, axis=1).sum()
        accuracy = correct / len(results_df)
        accuracies[ratio] = accuracy

    # Printing results for each ratio
    for ratio in thresholds:
        print(f"Ratio {ratio}: Threshold {thresholds[ratio]}, Accuracy {accuracies[ratio]}")

process_bookmia_data(dataset)
