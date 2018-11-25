import csv
import math

with open('C:/Users/David/Documents/GitHub/textClassification/result/pretrained_embedding (desktop)/test.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=';')
    rows = []
    for row in spamreader:
        rows = rows + [row]

    # pos_length = math.sqrt(sum([float(value)*float(value) for value in rows[1]]))
    # neg_length = math.sqrt(sum([float(value)*float(value) for value in rows[2]]))
    values = [float(pos_value)-float(neg_value) for pos_value,neg_value in zip(rows[1],rows[2])]
    # sum_pos = 0
    # sum_neg = 0
    # for value in values:
    #     if value < 0:
    #         sum_neg += value*value
    #     if value > 0:
    #         sum_pos += value*value

    # sum_pos = math.sqrt(sum_pos)
    # sum_neg = math.sqrt(sum_neg)
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
        
    

    