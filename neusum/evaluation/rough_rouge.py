STOP_WORDS = ['.', 'the', ',', 'to', 'of', 'a', 'and', 'in', "'s", 'was', 'for', 'that',
              '`', 'on', 'is', 'The', "'", 'with', 'said', ':', 'his', 'he',
              'at', 'as', 'it', 'I', 'from', 'have', 'has', 'by', '``', "''",
              'be', 'her', 'are', 'who', 'an', 'had', 'not', 'been', 'were', 'they',
              'their', 'after', 'she', 'but', '-RRB-', 'this', '-LRB-', 'will', '--',
              'which', "n't", 'It', 'when', 'up', 'out', 'one', 'about', 'more', 'He',
              '-', 'or', 'would', 'we', 'you', 'also', 'people', 'A', 'him',
              'can', 'into', 'two', 'told', 'than', 'all', 'do', 'time',
              'over', 'just', 'could', 'last', 'But', 'years', 'there',
              'first', 'its', 'them', 'so', 'year', 'In', 'before', 'being',
              'down', 'Mr', 'what', 'other', 'We', 'my', 'no', 'new', 'now',
              'did', 'like', 'home', 'left', 'some', 'if', 'because',
              'back', 'where', 'CNN', '#', 'only', 'She', 'against', 'while',
              'found', 'made', 'during', 'get', 'three', 'off', 'family', '?',
              'our', 'right', 'then', 'most', 'me', 'how', 'very', 'They',
              'says', 'around', '$', 'day', 'EST', 'This', 'through', 'any',
              'New', 'By', 'make', 'world', 'man', 'many', 'children', 'even',
              'way', 'week', 'since', 'take', 'life']


def _get_1gram_sets(highlights):
    """The input highlights should be a sequence of texts."""
    highlights = list(filter(lambda x: x != "", highlights.split(" ")))
    set_1gram = set(highlights)
    return set_1gram


def get_rouge_est_str_1gram(gold: str, pred: str) -> int:
    """
    Given two string, return the rouge1 F1.
    :param gold:
    :param pred:
    :return:
    """
    cand_1gram = _get_1gram_sets(pred)
    gold_1gram = _get_1gram_sets(gold)
    rouge_recall_1 = 0
    if len(gold_1gram) != 0:
        rouge_recall_1 = float(len(gold_1gram.intersection(cand_1gram))) / float(len(gold_1gram))
    rouge_pre_1 = 0
    if len(cand_1gram) != 0:
        rouge_pre_1 = float(len(gold_1gram.intersection(cand_1gram))) / float(len(cand_1gram))

    f1 = 0 if (rouge_recall_1 + rouge_pre_1 == 0) else 2 * (rouge_recall_1 * rouge_pre_1) / (
            rouge_recall_1 + rouge_pre_1)
    # print(rouge_recall_1, rouge_recall_2, rouge_recall_3, rouge_recall_4, rouge_recall_l, rouge_recall_average)
    return f1


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


def get_rouge_est_str_4gram(gold, pred):
    cand_1gram, cand_2gram, cand_3gram, cand_4gram = _get_ngram_sets(pred)
    gold_1gram, gold_2gram, gold_3gram, gold_4gram = _get_ngram_sets(gold)
    rouge_recall_1 = 0
    if len(gold_1gram) != 0:
        rouge_recall_1 = float(len(gold_1gram.intersection(cand_1gram))) / float(len(gold_1gram))
    rouge_recall_2 = 0
    if len(gold_2gram) != 0:
        rouge_recall_2 = float(len(gold_2gram.intersection(cand_2gram))) / float(len(gold_2gram))
    rouge_recall_3 = 0
    if len(gold_3gram) != 0:
        rouge_recall_3 = float(len(gold_3gram.intersection(cand_3gram))) / float(len(gold_3gram))
    rouge_recall_4 = 0
    if len(gold_4gram) != 0:
        rouge_recall_4 = float(len(gold_4gram.intersection(cand_4gram))) / float(len(gold_4gram))

    rouge_pre_1 = 0
    if len(cand_1gram) != 0:
        rouge_pre_1 = float(len(gold_1gram.intersection(cand_1gram))) / float(len(cand_1gram))

    rouge_pre_2 = 0
    if len(cand_2gram) != 0:
        rouge_pre_2 = float(len(gold_1gram.intersection(cand_2gram))) / float(len(cand_2gram))

    rouge_pre_3 = 0
    if len(cand_3gram) != 0:
        rouge_pre_3 = float(len(gold_3gram.intersection(cand_3gram))) / float(len(cand_3gram))

    rouge_pre_4 = 0
    if len(cand_4gram) != 0:
        rouge_pre_4 = float(len(gold_4gram.intersection(cand_4gram))) / float(len(cand_4gram))

    f1 = 0 if (rouge_recall_1 + rouge_pre_1 == 0) else 2 * (rouge_recall_1 * rouge_pre_1) / (
            rouge_recall_1 + rouge_pre_1)
    f2 = 0 if (rouge_recall_2 + rouge_pre_2 == 0) else 2 * (rouge_recall_2 * rouge_pre_2) / (
            rouge_recall_2 + rouge_pre_2)
    f3 = 0 if (rouge_recall_3 + rouge_pre_3 == 0) else 2 * (rouge_recall_3 * rouge_pre_3) / (
            rouge_recall_3 + rouge_pre_3)
    f4 = 0 if (rouge_recall_4 + rouge_pre_4 == 0) else 2 * (rouge_recall_4 * rouge_pre_4) / (
            rouge_recall_4 + rouge_pre_4)
    average_f_score = (f1 + f2 + f3 + f4)
    # print(rouge_recall_1, rouge_recall_2, rouge_recall_3, rouge_recall_4, rouge_recall_l, rouge_recall_average)
    return average_f_score


def _get_1gram_sets(highlights):
    highlights = list(filter(lambda x: x != "", highlights.split(" ")))

    fullen = len(highlights)

    set_1gram = set(map(lambda widx: str(highlights[widx]), range(fullen)))
    return set_1gram


def _get_2gram_sets(highlights):
    """The input highlights should be a sequence of texts."""
    highlights = list(filter(lambda x: x != "", highlights.split(" ")))

    fullen = len(highlights)

    set_1gram = set(map(lambda widx: str(highlights[widx]), range(fullen)))
    set_2gram = set(map(lambda widx: str(highlights[widx]) + "-" + str(highlights[widx + 1]), range(fullen - 1)))
    return set_1gram, set_2gram


def _get_ngram_sets(highlights):
    """The input highlights should be a sequence of texts."""
    highlights = list(filter(lambda x: x != "", highlights.split(" ")))

    fullen = len(highlights)

    set_1gram = set(map(lambda widx: str(highlights[widx]), range(fullen)))
    set_2gram = set(map(lambda widx: str(highlights[widx]) + "-" + str(highlights[widx + 1]), range(fullen - 1)))
    set_3gram = set(
        map(lambda widx: str(highlights[widx]) + "-" + str(highlights[widx + 1]) + "-" + str(highlights[widx + 2]),
            range(fullen - 2)))
    set_4gram = set(
        map(lambda widx: str(highlights[widx]) + "-" + str(highlights[widx + 1]) + "-" + str(
            highlights[widx + 2]) + "-" + str(highlights[widx + 3]),
            range(fullen - 3)))
    return set_1gram, set_2gram, set_3gram, set_4gram


def _get_lcs(a, b):
    lengths = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
    # read the substring out from the matrix
    result = []
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x - 1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y - 1]:
            y -= 1
        else:
            assert a[x - 1] == b[y - 1]
            result = [a[x - 1]] + result
            x -= 1
            y -= 1
    return len(result)


def get_rouge_est_str(gold, pred):
    cand_1gram, cand_2gram, cand_3gram, cand_4gram = _get_ngram_sets(pred)
    gold_1gram, gold_2gram, gold_3gram, gold_4gram = _get_ngram_sets(gold)
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

    # Get ROUGE-L
    len_lcs = _get_lcs(pred, gold)
    r = 0 if (len_lcs == 0) else (float(len_lcs) / len(pred))
    p = 0 if (len_lcs == 0) else (float(len_lcs) / len(gold))
    b = 0 if (r == 0) else (p / r)
    rouge_recall_l = 0 if (len_lcs == 0) else (((1 + (b * b)) * r * p) / (r + (b * b * p)))
    rouge_pre_l = 0 if (len_lcs == 0) else (((1 + (b * b)) * r * p) / (p + (b * b * r)))

    f1 = 0 if (rouge_recall_1 + rouge_pre_1 == 0) else 2 * (rouge_recall_1 * rouge_pre_1) / (
            rouge_recall_1 + rouge_pre_1)
    f2 = 0 if (rouge_recall_2 + rouge_pre_2 == 0) else 2 * (rouge_recall_2 * rouge_pre_2) / (
            rouge_recall_2 + rouge_pre_2)
    fl = 0 if rouge_pre_l + rouge_recall_l == 0 else 2 * (rouge_pre_l * rouge_recall_l) / (rouge_pre_l + rouge_recall_l)
    average_f_score = (f1 + f2 + fl) / 3
    # print(rouge_recall_1, rouge_recall_2, rouge_recall_3, rouge_recall_4, rouge_recall_l, rouge_recall_average)
    return average_f_score, f1, f2, fl
