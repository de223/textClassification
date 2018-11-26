import csv
import math

with open('C:/Users/David/Documents/GitHub/textClassification/result/pretrained_embedding (desktop)/test.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=';')
    rows = []
    for row in spamreader:
        rows = rows + [row]

    values = [float(pos_value)-float(neg_value) for pos_value,neg_value in zip(rows[1],rows[2])]

    length = math.sqrt(sum([value*value for value in values]))

    normalized_values = [value / length for value in values]
    number_of_words = len(rows[0])
    output_string = ""
    for i, value in enumerate(normalized_values):
        if(i < number_of_words):
            if value < 0:
                output_string += "\\colorbox{red!"+ str(int(-value*100))+ "}{"+ rows[0][i] + "} "
            else:
                if value > 0:
                    output_string += "\\colorbox{green!"+ str(int(value*100))+ "}{"+ rows[0][i] + "} "
                else:
                    output_string += "\\colorbox{white}{"+ rows[0][i] + "} "
        else:
            break
    if number_of_words >= i:
        for j in range(300,number_of_words):
            output_string += rows[0][j] + " "
            
    print(output_string)
        
    

    