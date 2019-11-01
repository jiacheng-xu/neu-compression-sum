def comp_num_seg_out_of_p_sent(_filtered_doc_list,
                               _num_edu,
                               _absas_read_str,
                               abs_as_read_list,
                               map_from_new_to_ori_idx):
    combs = []
    if len(_filtered_doc_list) < _num_edu:
        combs += list(range(len(_filtered_doc_list)))
    else:
        combs += list(itertools.combinations(range(len(_filtered_doc_list)), _num_edu))

    inp_doc = []
    for comb in combs:
        _tmp = assemble_doc_list_from_idx(_filtered_doc_list, comb)
        inp_doc.append('\n'.join(_tmp))
    inp_abs = [_absas_read_str] * len(combs)

    out = []
    for inp_d, inp_a in zip(inp_doc, inp_abs):
        o = get_rouge_est_str_2gram(inp_a, inp_d)
        out.append(o)
    # out_sort = sorted(out, key=itemgetter(2), reverse=True)
    sorted_idx = [i[0] for i in sorted(enumerate(out), key=lambda x: x[1][0], reverse=True)]

    # Write oracle to a string like: 0.4 0.3 0.4
    _comb_bag = {}
    for top_n_idx in sorted_idx[:top_k_combo]:
        n_comb = list(combs[top_n_idx])
        n_comb_original = [map_from_new_to_ori_idx[a] for a in n_comb]
        n_comb_original.sort()  # json label
        n_comb_original = [int(x) for x in n_comb_original]
        # print(n_comb_original)
        _tmp = assemble_doc_list_from_idx(_filtered_doc_list, combs[top_n_idx])

        score = rouge_protocol([[_tmp]], [[abs_as_read_list]])
        f1 = score['ROUGE-1-F']
        f2 = score['ROUGE-2-F']
        fl = score['ROUGE-L-F']
        f_avg = (f1 + f2 + fl) / 3
        _comb_bag[f_avg] = {"label": n_comb_original,
                            "R1": f1,
                            "R2": f2,
                            "RL": fl,
                            "R": f_avg,
                            "nlabel": _num_edu}
    # print(_comb_bag)
    # print(len(_comb_bag))
    best_key = sorted(_comb_bag.keys(), reverse=True)[0]
    rt_dict = {"nlabel": _num_edu,
               "data": _comb_bag,
               "best": _comb_bag[best_key]
               }
    return rt_dict



def random_compression(sent_del_spans,
                       abs_str: str,
                       num_of_compression) -> float:
    """
    Randomly drop num_of_compression compressions in sent_del_spans
    :param sent_del_spans:
    :param abs_str:
    :param num_of_compression:
    :return: rouge val after drop
    """
    texts = []
    compressions = []
    for unit in sent_del_spans:
        texts.append(unit[0].text)
        compressions.append(unit[1])
    sent_num = len(sent_del_spans)
    text_bits = [set(range(len(x))) for x in texts]
    list_of_compressions = []
    for sent_idx, comp in enumerate(compressions):
        for one_comp_option in comp:
            if one_comp_option['node'] != "BASELINE":
                list_of_compressions.append([sent_idx, one_comp_option])
    total_len = len(list_of_compressions)
    if num_of_compression < total_len:
        sample = random.sample(list_of_compressions, num_of_compression)
    else:
        sample = random.sample(list_of_compressions, total_len)

    for samp in sample:
        sent_target = samp[0]
        compression_to_try = samp[1]['selected_idx']

        before = text_bits[sent_target]
        after = set(before) - compression_to_try
        text_bits[sent_target] = after

    new_rouge = get_rouge_est_str_2gram(gold=abs_str, pred=
    assemble_text_and_bit(texts, text_bits))
    return new_rouge

# Compression of some sentences. Every sentence has at most C compressions. Compressions are not mutually exclusive.
# for every b in beam,
#   try all of the available deletion to increase the rouge value.
# until nothing available can improve rouge (improve thresold = + 2%)
def multiple_compression_sentence(sent_del_spans,
                                  abs_str: str, topk=10):
    finished_beam = []
    texts = []
    compressions = []
    for unit in sent_del_spans:
        texts.append(unit[0].text)
        compressions.append(unit[1])
    sent_num = len(sent_del_spans)
    text_bits = [set(range(len(x))) for x in texts]
    baseline_rouge = get_rouge_est_str_2gram(
        gold=abs_str, pred=assemble_text_and_bit(texts, text_bits))
    init_beam = {"compressions": compressions,
                 "text_bits": text_bits,
                 "done": []}
    B = [init_beam]

    while True:
        candidates = []
        for b in B:
            # b = [texts, compressions]

            current_rouge = get_rouge_est_str_2gram(
                gold=abs_str, pred=assemble_text_and_bit(texts, b["text_bits"]))

            flag, text_bit_list, val = [], [], []
            for sent in b['compressions']:
                flag.append([False for _ in range(len(sent))])
                text_bit_list.append([None for m in range(len(sent))])
                val.append([0 for m in range(len(sent))])
            # finish init

            for idx in range(sent_num):
                compression_of_sent = b['compressions'][idx]
                for jdx, compress_of_ith_sent in enumerate(compression_of_sent):
                    if b["compressions"][idx][jdx]["node"] == "BASELINE":
                        continue
                    tmp_text_bits = copy.deepcopy(b["text_bits"])

                    compression_to_try = b["compressions"][idx][jdx]['selected_idx']

                    # which sent
                    before = tmp_text_bits[idx]
                    after = set(before) - compression_to_try
                    tmp_text_bits[idx] = after
                    new_rouge = get_rouge_est_str_2gram(gold=abs_str, pred=
                    assemble_text_and_bit(texts, tmp_text_bits))

                    if new_rouge > current_rouge * 1.01:
                        flag[idx][jdx] = True
                        text_bit_list[idx][jdx] = tmp_text_bits
                        val[idx][jdx] = new_rouge
            # according to flag and vals, remove invalid compressions
            refresh_compression = [[] for _ in range(sent_num)]
            refresh_text_bits = [[] for _ in range(sent_num)]
            refresh_rouge = [[] for _ in range(sent_num)]
            for idx, ff in enumerate(flag):
                for jdx, f in enumerate(ff):
                    if f:
                        refresh_compression[idx].append(b["compressions"][idx][jdx])
                        refresh_text_bits[idx].append(text_bit_list[idx][jdx])
                        refresh_rouge[idx].append(val[idx][jdx])
            # now we have trimmed version of compression and it's value.

            for sent_idx in range(sent_num):
                comp_options = refresh_compression[sent_idx]
                _text_bits = refresh_text_bits[sent_idx]
                # _has_been_done = refresh_done[sent_idx]
                _has_been_done = b["done"]
                num_of_options = len(comp_options)
                for comp_idx in range(num_of_options):
                    this_comp_op = comp_options[comp_idx]
                    # done record
                    done_record = _has_been_done + [[sent_idx, this_comp_op]]
                    this_text_bit = _text_bits[comp_idx]
                    this_rouge = refresh_rouge[sent_idx][comp_idx]
                    if check_if_redundant(text_bit=this_text_bit, pool=candidates + finished_beam):
                        continue  # redundant!!!!!
                    copy_of_compression = copy.deepcopy(refresh_compression)
                    del copy_of_compression[sent_idx][comp_idx]
                    if check_if_empty_lists(copy_of_compression):
                        finished_beam.append({
                            "compressions": copy_of_compression,
                            "text_bits": this_text_bit,
                            "done": done_record,
                            "rouge": this_rouge
                        })
                    else:
                        candidates.append({
                            "compressions": copy_of_compression,
                            "text_bits": this_text_bit,
                            "done": done_record,
                            "rouge": this_rouge
                        })
                        finished_beam.append({
                            "compressions": copy_of_compression,
                            "text_bits": this_text_bit,
                            "done": done_record,
                            "rouge": this_rouge
                        })

        # how to end loop
        finished_beam = sorted(finished_beam, key=lambda x: x['rouge'], reverse=True)[:topk]
        B = candidates
        if B == []:
            break

    return finished_beam, baseline_rouge

