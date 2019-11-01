def convertion(singleoracle, doc, highlights):
    pass


def read_single_oracles(path_to_singleoracle, name=['cnn', 'dailymail']):
    with open(path_to_singleoracle, 'r') as fd:
        lines = fd.read().splitlines()
    l = ' '.join(lines)
    for n in name:
        l = l.replace('  ' + n, '\n' + n)
    l = l.split('\n')
    dict = {}
    for i in l:
        everything = i.split()
        sample_name = everything[0]
        sample_oracle = [int(x) for x in everything[1:]]
        sample_oracle_idx = [idx for idx, j in enumerate(sample_oracle) if j == 1]
        dict[sample_name] = sample_oracle_idx
    return dict


def read_multiple_oracles(path_to_mo, sent_idx_limit=30, sent_num_limit=3):
    pass


def read_doc_or_abs(path_to_doc,
                    path_to_vocab="/backup2/jcxu/exComp/data/vocab_list.txt", enable_span=True):
    with open(path_to_doc, 'r') as fd:
        lines = fd.read().splitlines()

    with open(path_to_vocab, 'r') as fd:
        vocab_lines = fd.read().splitlines()
    dict_vocab = []
    for vl in vocab_lines:
        dict_vocab.append(vl)

    all_doc = []
    _buff = []
    for l in lines:
        if l.startswith("cnn"):
            if _buff != []:
                all_doc.append(_buff)
            _buff = []
            _buff.append(l)
        elif len(l) >= 1:
            _buff.append(l)
        # if len(l) < 1 and len(_buff) > 1:
        #     all_doc.append(_buff)
        #     _buff = []
        # else:
        #     if len(l) >= 1:
        #         _buff.append(l)
    if _buff != []:
        all_doc.append(_buff)
    for d in all_doc:
        assert d[0].startswith('cnn')
    print(len(all_doc))
    print(all_doc[0])
    print(all_doc[-1])
    BAG = {}
    for d in all_doc:
        name = d[0]
        content = d[1:]
        span_info = []
        marker = 0
        _rt_doc = []

        for c in content:
            _seq = [dict_vocab[int(x)] for x in c.split()] + ["@@SS@@"]
            span_info.append(marker)
            span_info.append(marker + len(_seq) - 1)
            marker += len(_seq)
            _rt_doc += _seq
        _d = {"name": name, "span": span_info, "txt": _rt_doc}
        BAG[name] = _d
    return BAG


def merge_doc_abs_oracle(domain, dict_doc, dict_abs, dict_oracle):
    wt_bag = []
    for k, doc in dict_doc.items():
        try:
            oracle = dict_oracle[k]
            abst = dict_abs[k]

            doc_txt = ' '.join(doc['txt'])
            abs_txt = ' '.join(abst['txt'])
            span = ' '.join([str(x) for x in doc['span']])
            ora = ' '.join([str(x) for x in oracle])
            rt = '\t'.join([k, doc_txt, abs_txt, span, ora])
            wt_bag.append(rt)
        except KeyError:
            print("Key Error: {}".format(k))
    with open("/backup2/jcxu/data/cnn-lapata/" + domain + '.txt', 'w') as fd:
        fd.write('\n'.join(wt_bag))


if __name__ == '__main__':
    root = '/backup2/jcxu/exComp/data/data/preprocessed-input-directory'
    path_ora = root + "/cnn.test.label.singleoracle"
    oracle = read_single_oracles(path_ora)
    path_abs = root + "/cnn.test.highlights"
    abs_bag = read_doc_or_abs(path_abs)
    path_doc = root + "/cnn.test.doc"
    doc_bag = read_doc_or_abs(path_doc)
    # merge all
    merge_doc_abs_oracle('test', doc_bag, abs_bag, oracle)
