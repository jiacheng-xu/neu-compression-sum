from nltk.stem import porter

stemmer = porter.PorterStemmer()

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
              'New', 'By', 'make', 'world', 'man', 'many', 'even',
              'way', 'since', 'take']


# STOP_WORDS = ["."]

def _get_2gram_sets_keep_stop(highlights: str, stemming=True):
    """The input highlights should be a sequence of texts."""
    highlights = list(filter(lambda x: x != "", highlights.split(" ")))

    full_len = len(highlights)
    set_1gram = set(map(lambda widx: str(highlights[widx]), range(full_len)))

    set_2gram = set(map(lambda widx: str(highlights[widx]) + "-" + str(highlights[widx + 1]), range(full_len - 1)))

    if stemming:
        set_stem = set(map(lambda widx: str(stemmer.stem(highlights[widx]))
        if len(highlights[widx]) > 3 else str(highlights[widx]), range(full_len)))
    else:
        set_stem = set()
    return set_1gram, set_2gram, set_stem


def _get_2gram_sets_rm_stop(highlights: str, stemming=True):
    """The input highlights should be a sequence of texts."""
    highlights = list(filter(lambda x: x != "", highlights.split(" ")))
    highlights_no_stop = list(filter(lambda x: x not in STOP_WORDS, highlights))

    full_len = len(highlights)
    partial_len = len(highlights_no_stop)
    set_1gram = set(map(lambda widx: str(highlights_no_stop[widx]), range(partial_len)))

    set_2gram = set(map(lambda widx: str(highlights[widx]) + "-" + str(highlights[widx + 1]), range(full_len - 1)))

    if stemming:
        set_stem = set(map(lambda widx: str(stemmer.stem(highlights_no_stop[widx]))
        if len(highlights_no_stop[widx]) > 3 else str(highlights_no_stop[widx]), range(partial_len)))
    else:
        set_stem = set()
    return set_1gram, set_2gram, set_stem


def get_rouge_est_str_2gram_smart_kick_stop_words(gold: str, pred: str, stemming=True):
    gold_1, gold_2, gold_st = _get_2gram_sets_rm_stop(gold, stemming)
    pred_1, pred_2, pred_st = _get_2gram_sets_rm_stop(pred, stemming)
    rouge_recall_1 = 0
    if len(gold_1) != 0:
        rouge_recall_1 = float(len(gold_1.intersection(pred_1))) / float(len(gold_1))

    rouge_pre_1 = 0
    if len(pred_1) != 0:
        rouge_pre_1 = float(len(gold_1.intersection(pred_1))) / float(len(pred_1))

    rouge_recall_2 = 0
    if len(gold_2) != 0:
        rouge_recall_2 = float(len(gold_2.intersection(pred_2))) / float(len(gold_2))

    rouge_pre_2 = 0
    if len(pred_2) != 0:
        rouge_pre_2 = float(len(gold_2.intersection(pred_2))) / float(len(pred_2))

    f1 = 0 if (rouge_recall_1 + rouge_pre_1 == 0) else 2 * (rouge_recall_1 * rouge_pre_1) / (
            rouge_recall_1 + rouge_pre_1)
    f2 = 0 if (rouge_recall_2 + rouge_pre_2 == 0) else 2 * (rouge_recall_2 * rouge_pre_2) / (
            rouge_recall_2 + rouge_pre_2)
    if stemming:

        rouge_recall_st = 0
        if len(gold_st) != 0:
            rouge_recall_st = float(len(gold_st.intersection(pred_st))) / float(len(gold_st))

        rouge_pre_st = 0
        if len(pred_st) != 0:
            rouge_pre_st = float(len(gold_st.intersection(pred_st))) / float(len(pred_st))
        f_st = 0 if (rouge_recall_st + rouge_pre_st == 0) else 2 * (rouge_recall_st * rouge_pre_st) / (
                rouge_recall_st + rouge_pre_st)

        average_f_score = f1 * 0.25 + f2 * 0.5 + 0.25 * f_st
    else:
        average_f_score = (f1 + f2) / 2
        # average_f_score = (f1 )     # TODO
    return average_f_score


def get_rouge_est_str_2gram_smart(gold: str, pred: str, stemming=False):
    """
    If not stemming, return (f1 + f2) / 2
    If stemming, return (f1 + f2 + stem_f1) / 3
    All of the case remove the stop words.
    :param gold:
    :param pred:
    :param stemming:
    :return:
    """
    gold_1, gold_2, gold_st = _get_2gram_sets_keep_stop(gold, stemming)  # TODO
    pred_1, pred_2, pred_st = _get_2gram_sets_keep_stop(pred, stemming)  # TODO
    rouge_recall_1 = 0
    if len(gold_1) != 0:
        rouge_recall_1 = float(len(gold_1.intersection(pred_1))) / float(len(gold_1))

    rouge_pre_1 = 0
    if len(pred_1) != 0:
        rouge_pre_1 = float(len(gold_1.intersection(pred_1))) / float(len(pred_1))

    rouge_recall_2 = 0
    if len(gold_2) != 0:
        rouge_recall_2 = float(len(gold_2.intersection(pred_2))) / float(len(gold_2))

    rouge_pre_2 = 0
    if len(pred_2) != 0:
        rouge_pre_2 = float(len(gold_2.intersection(pred_2))) / float(len(pred_2))

    f1 = 0 if (rouge_recall_1 + rouge_pre_1 == 0) else 2 * (rouge_recall_1 * rouge_pre_1) / (
            rouge_recall_1 + rouge_pre_1)
    f2 = 0 if (rouge_recall_2 + rouge_pre_2 == 0) else 2 * (rouge_recall_2 * rouge_pre_2) / (
            rouge_recall_2 + rouge_pre_2)
    if stemming:

        rouge_recall_st = 0
        if len(gold_st) != 0:
            rouge_recall_st = float(len(gold_st.intersection(pred_st))) / float(len(gold_st))

        rouge_pre_st = 0
        if len(pred_st) != 0:
            rouge_pre_st = float(len(gold_st.intersection(pred_st))) / float(len(pred_st))
        f_st = 0 if (rouge_recall_st + rouge_pre_st == 0) else 2 * (rouge_recall_st * rouge_pre_st) / (
                rouge_recall_st + rouge_pre_st)

        average_f_score = f1 * 0.25 + f2 * 0.5 + 0.25 * f_st
    else:
        average_f_score = (f1 + f2) / 2
        # average_f_score = (f1 )     # TODO
    return average_f_score


if __name__ == '__main__':
    gold = "Artist and journalist Alison Nastasi put together the portrait collection. Also features images of Picasso, Frida Kahlo, and John Lennon. Reveals quaint personality traits shared between artists and their felines."
    sent = ["Too often cats can be overlooked. ",
            "Artists And Their Cats, is putting felines back on the map. ",
            "Philadelphia-based artist and journalist Alison Nastasi has collated a collection of intimate portraits featuring artists their . ",
            "Philadelphia-based artist and journalist Alison Nastasi has collated a collection of intimate portraits featuring artists with their . Spanish surrealist painter Salvador Dali poses with his cat Babou.",
            "Wanda Hazel Gág, lovingly looks at her cat in one of the intimate portraits.",
            "the book allows an intimate insight into the private lives of many great artists.",
            "The images see the artists in rarely-seen settings or posing in a relaxed fashion with their beloved pets."
            ]
    for idx in range(1, len(sent) + 1):
        for jdx in range(idx + 1, len(sent) + 1):
            for kdx in range(jdx + 1, len(sent) + 1):
                score = get_rouge_est_str_2gram_smart(gold=gold,
                                                      pred=sent[idx - 1] + " " + sent[jdx - 1] + " " + sent[kdx - 1])
                print("{},{},{}\t{}".format(idx, jdx, kdx, score))
    # [print(get_rouge_est_str_2gram_smart(gold=gold,pred=sx)) for sx in sent]
    # pred = "Philadelphia-based artist and journalist Alison Nastasi has collated a collection of intimate portraits featuring well known artists with their furry friends."
    # ori = get_rouge_est_str_2gram_smart(gold=gold, pred=pred)
    # print(ori)
    # pred_rmp = "artist and journalist Alison Nastasi has collated a collection of intimate portraits featuring well known artists with their furry friends."
    # ori = get_rouge_est_str_2gram_smart(gold=gold, pred=pred_rmp)
    # print(ori)
    # pred_rmintimate = "Philadelphia-based artist and journalist Alison Nastasi has collated a collection of portraits featuring well known artists with their furry friends."
    # ori = get_rouge_est_str_2gram_smart(gold=gold, pred=pred_rmintimate)
    # print(ori)
    # pred_rm_feat = "Philadelphia-based artist and journalist Alison Nastasi has collated a collection of intimate portraits."
    # ori = get_rouge_est_str_2gram_smart(gold=gold, pred=pred_rm_feat)
    # print(ori)
    #
    # pred_rm_wellknow = "Philadelphia-based artist and journalist Alison Nastasi has collated a collection of intimate portraits featuring artists with their furry friends."
    # ori = get_rouge_est_str_2gram_smart(gold=gold, pred=pred_rm_wellknow)
    # print(ori)
