import binoculars
from pandas import pd

observer_model, performer_model = binoculars.create_models()
binoculars_obj = binoculars.Binoculars(observer_model, performer_model, "accuracy", 512)

file_path = '../data/vulnerable_texts.csv'
df = pd.read_csv(file_path)

for i in range(len(df['Text'])):
  print(df['Text Name'][i])
  binoculars_score, min_k_result = binoculars_obj.compute_score_and_min_k(df['Text'][i], 0.05, 3.809450257187945e-08)
  predictions = binoculars.predict(df['Text'][i])
  print("Memorized?: " + str(min_k_result[0]))
  print("Binoculars Score: "+str(binoculars_score[0]))
  print("Prediction: "+str(predictions[0]))
  print()

