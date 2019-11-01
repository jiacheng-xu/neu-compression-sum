import pickle, os

if __name__ == '__main__':
    # os.chdir('dm_more')

    # f = "0.427,0.191,0.387-dmTrue10.0-1True3-10397-cp_0.8"
    f = '0.327,0.122,0.290-cnnTrue1.0-1True3-1093-cp_0.5'
    with open(f, 'rb') as fd:
        x = pickle.load(fd)
    preds = x['pred']
    oris = x['ori']
    refs = x['ref']

    for p, o, r in zip(preds, oris, refs):

        p_words = ' '.join(p)
        p_words_list = p_words.split(' ')
        p_words_set = set(p_words_list)
        # for _p in p:
        #     p_words += ' '+ _p

        o_words = ' '.join(o)
        o_words_list = o_words.split(' ')
        o_words_set = set(o_words_list)
        # for _o in o:
        #     o_words += _o

        r_words = ' '.join(r[0])
        r_words_list = r_words.split(' ')
        r_words_set = set(r_words_list)
        # for _r in r:
        #     r_words += _r
        # if len(r_words_set.intersection(p_words_set)) >= len(r_words_set.intersection(o_words_set)) and len(
        #         p_words_list) + 10 < len(o_words_list) and len(r_words_set.intersection(p_words_set)) > 5:
        if True:
            print(p_words)
            print(o_words)
            print(r_words)
            print('-' * 20)
