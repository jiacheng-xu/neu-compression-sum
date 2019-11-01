from nltk.stem import porter

stemmer = porter.PorterStemmer()


def _get_2gram_sets(highlights):
    """The input highlights should be a sequence of texts."""
    highlights = list(filter(lambda x: x != "", highlights.split(" ")))

    fullen = len(highlights)

    set_1gram = set(map(lambda widx: str(highlights[widx]), range(fullen)))
    set_2gram = set(map(lambda widx: str(highlights[widx]) + "-" + str(highlights[widx + 1]), range(fullen - 1)))
    return set_1gram, set_2gram


def get_rouge_est_str_2gram(gold, pred) -> float:
    cand_1gram, cand_2gram = _get_2gram_sets(pred)
    gold_1gram, gold_2gram = _get_2gram_sets(gold)
    rouge_recall_1 = 0
    if len(gold_1gram) != 0:
        rouge_recall_1 = float(len(gold_1gram.intersection(cand_1gram))) / float(len(gold_1gram))
    rouge_recall_2 = 0
    if len(gold_2gram) != 0:
        rouge_recall_2 = float(len(gold_2gram.intersection(cand_2gram))) / float(len(gold_2gram))

    rouge_pre_1 = 0
    if len(cand_1gram) != 0:
        rouge_pre_1 = float(len(gold_1gram.intersection(cand_1gram))) / float(len(cand_1gram))

    rouge_pre_2 = 0
    if len(cand_2gram) != 0:
        rouge_pre_2 = float(len(gold_1gram.intersection(cand_1gram))) / float(len(cand_2gram))

    f1 = 0 if (rouge_recall_1 + rouge_pre_1 == 0) else 2 * (rouge_recall_1 * rouge_pre_1) / (
            rouge_recall_1 + rouge_pre_1)
    f2 = 0 if (rouge_recall_2 + rouge_pre_2 == 0) else 2 * (rouge_recall_2 * rouge_pre_2) / (
            rouge_recall_2 + rouge_pre_2)
    average_f_score = (f1 + f2) / 2

    # print(rouge_recall_1, rouge_recall_2, rouge_recall_3, rouge_recall_4, rouge_recall_l, rouge_recall_average)
    return average_f_score


from typing import List

CONTENT_LIB = ['.', 'the', ',', 'to', 'of', 'a', 'and', 'in', "'s", 'was', 'for', 'that', '`', 'on',
               'is', 'The', "'", 'with', 'said', ':', 'his', 'he', 'at', 'as', 'it', 'I', 'from',
               'have', 'has', 'by', '``', "''", 'be', 'her', 'are', 'who', 'an', 'had', 'not', 'been',
               'were', 'they', 'their', 'after', 'she', 'but', 'this', 'will', '--', 'which', "n't",
               'It', 'when', 'up', 'out', 'one', 'about', 'she', 'before', 'up', 'such', 'it', 'mustn',
               "mustn't", 'shouldn', 'most', 'couldn', 'more', 'does',
               'for', 'needn', 'through', 'once', 'there', 'after', 'these', 'those']
CONTENT_LIB = list(set(CONTENT_LIB))


def length_compensation(doc, abs: List) -> str:
    if type(doc) is str:
        doc = doc.split(" ")
    l_abs = len(abs)
    l_doc = len(doc)
    if l_abs > l_doc:
        gap = l_abs - l_doc
        backup = CONTENT_LIB[:gap]
    else:
        backup = []
    doc = doc + backup
    return " ".join(doc)


def plain():
    pass


import re


def tok_preprocess(text: List):
    output = []
    output_stem = []
    for e in text:
        o = re.sub(r"[^a-zA-Z0-9]+", " ", e)
        tokens = re.split(r"\s+", o)
        tokens = [stemmer.stem(x) if len(x) > 3 else x for x in tokens]
        o_stem = " ".join(tokens)
        output.append(o)
        output_stem.append(o_stem)
    return output, output_stem


# text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
# options available
# stemming   lowercase   rouge 1   rouge 2 rouge 1+2
with open("/backup3/jcxu/data/cnndm_vocab/tokens.txt", 'r') as fd:
    lines = fd.read().splitlines()
stop = lines[:300]
print(stop)



if __name__ == '__main__':


    examples = [
        "President Donald Trump outlined a plan to end the government shutdown on Saturday, "
        "offering congressional Democrats three years of legislative relief for 700,000 DACA recipients — "
        "including protection from deportation — and an extension of legal residence for people living in the country under "
        "'Temporary Protective Status' designations.",
        "Caydee Denney and John Coughlin, of the United States, perform "
        "during their Pairs Short Program during the ISU Figure Skating "
        "Eric Bompard Trophy at Bercy arena in Paris in 2013",
        "Mitt Romney has backed the president's push for a border wall saying he 'doesn't understand' Nancy Pelosi's position. The Republican has been a harsh critic Donald Trump, right",
        "-LRB- CNN -RRB- He fucks the whole universe and that's all.",
        "If he is at the but by which night could be the government shutdown on Saturday",  # if the whole sent is kind of stop words
    ]

    examples_compression = [[
        "President Donald Trump outlined a plan , "
        "offering congressional Democrats three years of legislative relief for 700,000 DACA recipients "
        "— including protection from deportation — and an extension of legal residence for people living in the country under "
        "'Temporary Protective Status' designations.",  # very bad
        "President outlined a plan to end the government shutdown on Saturday, "
        "offering congressional Democrats three years of legislative relief for 700,000 DACA recipients "
        "— including protection from deportation — and an extension of legal residence for people living in the country under "
        "'Temporary Protective Status' designations.",  # slightly good remove Donald Trump
        "President Donald Trump outlined a plan to end the government shutdown on Saturday, "
        "offering congressional Democrats three years of legislative relief for 700,000 DACA recipients "
        "— including protection from deportation — and an extension "
        "'Temporary Protective Status' designations.",
        # very good remove  of legal residence for people living in the country

    ], ["Caydee Denney and John Coughlin, of the United States, perform  "
        "Short Program during the ISU Figure Skating Eric Bompard Trophy at Bercy arena in 2013",  # bad

        "Caydee Denney and John Coughlin perform during their Pairs ,"
        "Short Program during the ISU Figure Skating Eric Bompard Trophy at Bercy arena in Paris in 2013",
        # slight good
        "Caydee Denney and John Coughlin, of the United States, perform during their Pairs "
        "Short Program Figure Skating Eric Bompard Trophy in Paris"  # very good
        ],
        [
            "Mitt Romney has backed the president's push for a border wall saying he 'doesn't understand' Nancy Pelosi's position. The Republican has been a critic Donald Trump, right",
            # bad
            "Romney has backed the president's push for a border wall saying he 'doesn't understand' Nancy Pelosi's position. The Republican has been a harsh critic Donald Trump, right",
            "Mitt Romney has backed the president's push for a border wall. The Republican has been a harsh critic Donald Trump",
        ],
    [
        "-LRB- CNN -RRB- He fucks the whole  and that's all.",
        " He fucks the whole universe and that's all.",
        " He fucks the whole universe ",
    ],
        [
            "If he is at the but by which night could be the government shutdown",
            " but by which night could be the government shutdown on Saturday",
            " which night could be the government shutdown on Saturday",
        ]
    ]

    abs = [
        "President detailed his latest offer to end the shutdown from the Diplomatic Reception Room of the "
        "White House on Saturday afternoon. He offered three years of legislative relief for 700,000 DACA recipients "
        "— including protection from deportation. Also offered a three-year extension for Temporary Protective Status "
        "protectees including those from El Salvador, Haiti, Honduras and Nicaragua. Wants $5.7 billion for the 'strategic deployment of physical barriers, or a wall,' that he will use to put 'steel barriers in high priority locations'.",

        "Pairs skating champion John Coughlin took his own life on Friday, "
        "his sister said. He was suspended by US Figure Skating on Thursday "
        "for unspecified cause. Coughlin won national pairs championships with two partners in 2011 and 2012",
        "The Republican has been a harsh critic of the president in the past . Said Trump's conduct had 'not risen to the mantle of the office'. But he questioned why the Democratic House Speaker won't agree to 'another few miles' of barriers on the U.S.-Mexico border",
        "His fucking the whole universe.",
        " the government shutdown on Saturday",
    ]
    ex, ex_stem = tok_preprocess(examples)
    a, a_stem = tok_preprocess(abs)
    from neusum.evaluation.smart_approx_rouge import get_rouge_est_str_2gram_smart
    print("Policy 1: Plain")
    for idx, compression in enumerate(examples_compression):
        compress, compress_stem = tok_preprocess(compression)
        val = [get_rouge_est_str_2gram_smart(a[idx], x) for x in compress]
        baseline = get_rouge_est_str_2gram_smart(a[idx], ex[idx])
        print("Val: {}".format(val))
        print("Ratio: {}".format([v / baseline for v in val]))

    print("Policy 1: Plain stem")
    for idx, compression in enumerate(examples_compression):
        compress, compress_stem = tok_preprocess(compression)
        val = [get_rouge_est_str_2gram_smart(a_stem[idx], x) for x in compress_stem]
        baseline = get_rouge_est_str_2gram_smart(a_stem[idx], ex_stem[idx])
        print("Val: {}".format(val))
        print("Ratio: {}".format([v / baseline for v in val]))

    """
    print("Policy 2: length compensation")
    for idx, compression in enumerate(examples_compression):
        compress, compress_stem = tok_preprocess(compression)
        new_compress = []
        for c in compress:
            len_compensat_inp = length_compensation(doc=c, abs=a[idx].split(" "))
            new_compress.append(len_compensat_inp)
        val = [get_rouge_est_str_2gram_rm_stop(a[idx], x) for x in new_compress]
        baseline = get_rouge_est_str_2gram_rm_stop(a[idx], length_compensation(doc=ex[idx], abs=a[idx].split(" "))  )
        print("Val: {}".format(val))
        print("Ratio: {}".format([v / baseline for v in val]))

    print("Policy 3: length compensation + useful stop words")
    for idx, compression in enumerate(examples_compression):
        compress, compress_stem = tok_preprocess(compression)
        for c in compress:
            len_compensat_inp = length_compensation(doc=c, abs=a[idx].split(" "))

    print("Policy 3: plain + length discount")
    """
