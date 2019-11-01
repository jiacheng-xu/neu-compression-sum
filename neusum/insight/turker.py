if __name__ == '__main__':
    num_of_examples = 8
    template = "<li><p>Please rate this piece of text in terms of " \
               "<u><strong>grammaticality</strong></u></p><p>" \
               "${text_{}}</p><p>Not grammatical (1) <===> Grammatical (10)<crowd-slider " \
               "name=\"rate_{}\" min=\"0\" max=\"10\" required pin></crowd-slider></p></li>"
    for i in range(num_of_examples):
        print("<li><p>Please rate this piece of text in terms of " \
              "<u><strong>grammaticality</strong></u></p><p>" \
              "${text_" + str(i) + "}</p><p>Not grammatical (1) <===> Grammatical (10)<crowd-slider " \
                                   "name=\"rate_" + str(
            i) + "\" min=\"0\" max=\"10\" required pin></crowd-slider></p></li>")
    for i in range(num_of_examples):
        print("text_" + str(i) + "\t" + "rate_" + str(i))
