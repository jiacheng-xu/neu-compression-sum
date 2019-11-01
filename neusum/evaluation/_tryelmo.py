from allennlp.commands.elmo import ElmoEmbedder
sent_pair = ["Top trade negotiators from China and the United States",
             "Top representatives from China and the US",
             "US President Donald Trump and his Chinese counterpart Xi Jinping"]

# sent = ["Top representatives from China and the US"]

elmo = ElmoEmbedder()
# tokens = ["I", "ate", "an", "apple", "for", "breakfast"]


# assert (len(vectors) == 3)  # one for each layer in the ELMo output
# assert (len(vectors[0]) == len(tokens))  # the vector elements correspond with the input tokens

import scipy
# vectors = elmo.embed_sentence(tokens)
# vectors2 = elmo.embed_sentence(["I", "ate", "a", "carrot", "for", "breakfast"])
# print()scipy.spatial.distance.cosine(vectors[2][3],
#                               vectors2[2][3])  # cosine distance between "apple" and "carrot" in the last layer

# sents = [
#     "Obese patients will be sent on cookery courses and to Zumba fitness classes by their GPs as part of a drive to tackle diabetes.",
#     "Family doctors are being encouraged to target patients most at risk, including those who are seriously overweight.",
#     "Under the plans, GPs will refer patients to gym classes such as Zumba, aerobics or spinning – intensive cycling – funded by the NHS.",
#     "Under the plans, GPs will refer patients to gym classes such as Zumba (pictured), aerobics or spinning – intensive cycling – funded by the NHS. People will also be advised to attend cookery courses",
#     "People will also be advised to attend cooking sessions teaching them how to poach, boil and grill food instead of frying it. ",
#     "The initiative extends to overweight NHS staff, who will be encouraged to take dieting classes held at hospitals in order to set a good example to patients. Around 3.8million Britons have diabetes. ",
#     "The new scheme will be announced today by NHS England chief executive Simon Stevens, pictured",
#     "The figure has doubled in 20 years, mainly due to obesity. The new scheme will be announced today by NHS England chief executive Simon Stevens.",
#     "In a speech at a London conference hosted by the charity Diabetes UK, he will say: ‘It’s time for the NHS to start practising what we preach. For over a decade we’ve known that obesity prevention cuts diabetes and saves lives.’"
# ]

# sents = ["eighty bucks", "80 dollars", "$ 80" ]

bag = []
for sidx, s in enumerate(sent_pair):
    s_tokens = s.split(" ")
    s_vec = elmo.embed_sentence(s_tokens)
    print(s_vec.shape)

    bag.append(s_vec[0][3])

output = [[] for _ in range(len(bag))]
for sidx, s in enumerate(sent_pair):
    for jdx, j in enumerate(sent_pair):

        output[sidx] += [scipy.spatial.distance.cosine(bag[sidx], bag[jdx])]

print(output)