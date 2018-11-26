import csv
import math

with open('./result/pretrained_embedding (desktop)/test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    rows = []
    for row in reader:
        rows = rows + [row]

    values = [float(pos_value)-float(neg_value) for pos_value,neg_value in zip(rows[1],rows[2])]
    length = math.sqrt(sum([value*value for value in values]))

    normalized_values = [value / length for value in values]
    number_of_words = len(rows[0])
    output_string = ""
    top_1 = [0,0]
    top_2 = [0,0]
    top_3 = [0,0]
    low_1 = [0,0]
    low_2 = [0,0]
    low_3 = [0,0]
    for i, value in enumerate(normalized_values):
        if(i < number_of_words):
            if value > 0:
                if value > top_1[1]:
                    top_3 = top_2
                    top_2 = top_1
                    top_1 = [i,value]
                elif value > top_2[1]:
                    top_3 = top_2
                    top_2 = [i,value]
                elif value > top_3[1]:
                    top_3 = [i,value]
            elif value < 0:
                if value < low_1[1]:
                    low_3 = low_2
                    low_2 = low_1
                    low_1 = [i,value]
                elif value < low_2[1]:
                    low_3 = low_2
                    low_2 = [i,value]
                elif value < low_3[1]:
                    low_3 = [i,value]
        else:
            break


    print("\\begin{multicols}{2}")
    print("Top positive words:")
    print("\\begin{enumerate}")
    print("\\item "+rows[0][top_1[0]] + ": " +  str(top_1[1]))
    print("\\item "+rows[0][top_2[0]] + ": " +  str(top_2[1]))
    print("\\item "+rows[0][top_3[0]] + ": " +  str(top_3[1]))
    print("\\end{enumerate}")
    print("Top negative words:")
    print("\\begin{enumerate}")
    print("\\item "+rows[0][low_1[0]] + ": " +  str(low_1[1]))
    print("\\item "+rows[0][low_2[0]] + ": " +  str(low_2[1]))
    print("\\item "+rows[0][low_3[0]] + ": " +  str(low_3[1]))
    print("\\end{enumerate}")
    print("\\end{multicols}")

    