import csv

input_string = """
Text Name: Gettysburg Address
Binoculars Score: 0.7724359035491943
Mink Score: 1.2665987014770508e-07
Text Name: MIT License
Binoculars Score: 0.918749988079071
Mink Score: 8.521601557731628e-08
Text Name: US Constitution Snippet
Binoculars Score: 0.8442028760910034
Mink Score: 1.4319084584712982e-08
Text Name: Boilerplate Terms of Service
Binoculars Score: 0.8282828330993652
Mink Score: 8.67992639541626e-07
Text Name: Bill of Rights
Binoculars Score: 0.7285714149475098
Mink Score: 6.845220923423767e-08
Text Name: Apache License
Binoculars Score: 0.8779527544975281
Mink Score: 2.9802322387695312e-08
Text Name: Boilerplate EULA Warranty Agreement
Binoculars Score: 0.7581395506858826
Mink Score: 1.0654330253601074e-06
Text Name: Declaration of Independence
Binoculars Score: 0.7142857313156128
Mink Score: 9.266659617424011e-08
Text Name: Introduction to the Magna Carta
Binoculars Score: 0.9078013896942139
Mink Score: 4.4517219066619873e-07
Text Name: Introduction to the Emancipation Proclamation
Binoculars Score: 0.803108811378479
Mink Score: 9.73232090473175e-08
Text Name: SeenMIA
Binoculars Score: 0.9764832234850117
Mink Score: 2.692891832660226e-07
Text Name: UnseenMIA
Binoculars Score: 1.005487576607735
Mink Score: 3.3538127618451273e-07
"""

# Split the input string by lines and then by ": "
data = [line.split(": ") for line in input_string.strip().split("\n")]

# Multiply all Mink scores by 1e6
for item in data:
    if item[0] == "Mink Score":
        item[1] = str(float(item[1]) * 1e9)

# Write the data to a CSV file
with open("scores.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Text Name", "Binoculars Score", "Mink Score"])
    for i in range(0, len(data), 3):
        writer.writerow([data[i][1], data[i+1][1], data[i+2][1]])
