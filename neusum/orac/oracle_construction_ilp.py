from neusum.service.basic_service import clear_dir
from neusum.orac.util import *
from neusum.evaluation.smart_approx_rouge import get_rouge_est_str_2gram_smart_kick_stop_words
from neusum.data.convert_json_to_pkl import convert_json_file_to_pkl_dump
from neusum.data.check_snlp_tree import find_deletable_span_rule_based_updated


def read_file(fpath) -> List[str]:
    with open(fpath, 'r') as fd:
        output = fd.read().splitlines()
    return output


def default_sent_as_sos():
    dft = {"sidx": 0, 'eidx': 1, 'node': "BASELINE", 'rouge': 0, 'selected_idx': []}
    # <SOS> Used for pred the end of decoding
    # sent_sos_dict = SentDataWithOracle(token=["<SOS>"],del_span= [dft],single_del=[dft],single_del_best=dft)
    sent_sos_dict = {'token': ["<SOS>"],
                     'del_span': [dft], 'single_del': [dft],
                     'single_del_best': dft}
    return sent_sos_dict


import random


def process_one_sentence(sent_parse: TreeNode, abs_str: str
                         ) -> List:
    """
    Given the parse tree of a sentence and the gold summary,
    get compression options and corresponding rouge value after performing this compression.
    node: BASELINE  =  operation: delete last token; rouge: the whole sentence. trick for convinience
    :param sent_parse:
    :param abs_str:
    :return: List[[sent_tree: TreeNode, rt_del_spans: List[{'node','rouge','selected_idx', 'ratio'}, {}], baselinve_rouge: float]]
    """
    abs_list = abs_str.split(" ")
    sent_tree = read_single_parse_tree(sent_parse)
    tree_len = len(sent_tree.text)
    abs_len = len(abs_str.split(" "))
    # len_compensat_sent = length_compensation(doc=sent_tree.text, abs=abs_list)
    sent_str = " ".join(sent_tree.text)
    rt_del_spans = []
    del_spans = find_deletable_span_rule_based_updated(sent_tree, root_len=tree_len, parent=None, grand_parent=None)
    baseline_rouge = get_rouge_est_str_2gram_smart_kick_stop_words(
        gold=abs_str, pred=sent_str)
    # new_baseline_rouge = folding_rouge(abs_list, sent_str)
    if baseline_rouge < 0.03:
        useless_baseline = True
    else:
        useless_baseline = False

    for del_sp in del_spans:
        if len(del_sp['selected_idx']) < 1:
            continue
        full_set = set(range(len(sent_tree.text)))
        remain_idx = list(full_set - del_sp['selected_idx'])
        remain_idx.sort()
        _txt = " ".join([sent_tree.text[idx] for idx in remain_idx])
        # _txt = length_compensation(_txt, abs=abs_list)
        _rouge = get_rouge_est_str_2gram_smart_kick_stop_words(gold=abs_str, pred=_txt)

        rt_del_spans.append(
            {'node': del_sp["node"],
             'rouge': _rouge,
             'selected_idx': list(del_sp["selected_idx"]),
             'ratio': _rouge / baseline_rouge if not useless_baseline else 1.0,
             'label': -1}
        )
    rt_del_spans.append({
        'node': "BASELINE",
        'rouge': get_rouge_est_str_2gram_smart_kick_stop_words(
            gold=abs_str, pred=sent_str),
        'selected_idx': [tree_len - 1],
        'ratio': 0.0,
        'label': -1})
    return [sent_tree, rt_del_spans, baseline_rouge]


def get_document_parse_tree_and_str(inp: List[str]) -> (List[TreeNode], List[str]):
    tree_bag, str_bag = [], []
    for sent in inp:
        out = read_single_parse_tree(sent)
        tree_bag.append(out)
        s = " ".join(out.text)
        str_bag.append(s)
    return tree_bag, str_bag


def read_file_no_splitlines(fpath):
    with open(fpath, 'r') as fd:
        output = fd.read()
    return output


from depricated.ILP.cvxpy_ilp import ILP_protocol, ILP_protocol_w_compression


def process_one_example(path_read: str, path_write,
                        fname_without_suffix: str, max_sent=40,
                        data_name: str = 'dm'):
    """
    Given one snlp parse output (parse trees of one document),
    produce json format data for production.
    :param path_read:
    :param path_write:
    :param fname_without_suffix:
    :param max_sent:
    :return:
    """
    doc_file = os.path.join(path_read, fname_without_suffix + ".doc.json")
    abs_file = os.path.join(path_read, fname_without_suffix + ".abs.json")
    if (not os.path.isfile(doc_file)) or (not os.path.isfile(abs_file)):
        raise TypeError
    try:
        doc_dict = json.loads(read_file_no_splitlines(doc_file))
        abs_dict = json.loads(read_file_no_splitlines(abs_file))
    except json.decoder.JSONDecodeError:
        print("forget it?")
        return

    doc_parse = extract_parse(doc_dict)
    abs_token, abs_str = extract_tokens(abs_dict)
    doc_token, doc_str = extract_tokens(doc_dict)
    sent_packs = []

    # filter doc
    cnt = 0
    _formal_doc_parse = []
    _formal_doc_token = []
    for doc_idx, doc_t in enumerate(doc_token):
        l = len(doc_t)
        if l > 3 and l < 70:  # stat shows l>70+ is  64 out of 26k
            _formal_doc_parse.append(doc_parse[doc_idx])
            _formal_doc_token.append(doc_t)
            cnt += 1
        if cnt >= max_sent:
            break
    doc_parse = _formal_doc_parse
    doc_token = _formal_doc_token

    if len(doc_parse) <2:
        print("forget {}".format(fname_without_suffix))
        return
    # comp compression options and deletion based on every single sentence
    _rt_sentences = []
    for idx, x in enumerate(doc_parse):
        output = process_one_sentence(x, abs_str)
        _rt_sentences.append(output)
    # _rt_sentences = [process_one_sentence(x, abs_str,context_sent=) for x in doc_parse]
    # sent_tree, rt_del_spans, baselinve_rouge

    for rt_sent in _rt_sentences:
        _tmp = {
            "token": rt_sent[0].text,
            "del_span": rt_sent[1],
            "baseline": rt_sent[2]
            # "tree": rt_sent[0]
        }
        sent_packs.append(_tmp)

    # comp document level sentence oracle
    doc_list = [" ".join(x['token']) for x in sent_packs]

    # ILP for compression oracle: both sent label and compression label
    ILP_protocol_w_compression(reference_summary=abs_str, sent_units=doc_list, compression=sent_packs)

    # ILP for extractive oracle: only sent label
    sent_ora_json = ILP_protocol(reference_summary=abs_str,
                                 sent_units=doc_list)

    if random.random() < 0.001:
        print(sent_ora_json)


        # pass
    # # comp document level delete-one sentence oracle
    # doc_list_trimmed_for_oracle = []
    # for x in sent_packs:
    #     _del_span = x['del_span']
    #     _del_span.sort(key=lambda y: y['rouge'], reverse=True)
    #     try:
    #         del_idx = _del_span[0]['selected_idx']
    #         selected_set = list(set(list(range(len(x['token'])))) - set(del_idx))
    #         selected_set.sort()
    #     except IndexError:
    #         selected_set = list(range(len(x['token'])))
    #     doc_list_trimmed_for_oracle.append(" ".join([x['token'][kdx] for kdx in selected_set]))
    # trim_ora_json = comp_document_oracle(doc_list_trimmed_for_oracle, abs_str)
    rt = {}
    rt["name"] = fname_without_suffix
    rt['part'] = data_name
    # span_pairs = gen_span_segmentation([x['token'] for x in rt_sentences])
    rt["abs"] = abs_str
    rt["abs_list"] = abs_token
    rt["doc"] = " ".join(doc_list)
    rt["doc_list"] = [x['token'] for x in sent_packs]
    rt["sentences"] = sent_packs

    rt["sent_oracle"] = sent_ora_json   # TODO
    rt["non_compression_sent_oracle"] = sent_ora_json
    json_rt = json.dumps(rt)
    with open(os.path.join(path_write, fname_without_suffix + '.data'), 'w') as fd:
        fd.write(json_rt)


def create_oracles_ilp(dataname, path_read, path_wt_distributed):
    """
    Create Oracles, write individual json files to the disk.
    If CNNDM,
    :return:
    """

    files = [i.split(".")[0] for i in os.listdir(path_read) if i.endswith('.doc.json')]
    # files = files[:100]
    total_num = len(files)
    cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cnt)

    pool.starmap(process_one_example, zip([path_read] * total_num,
                                          [path_wt_distributed] * total_num,
                                          files,
                                          [40] * total_num,
                                          [dataname] * total_num))
    pool.close()
    pool.join()
    print("finish creating oracles, and write down them to distributed folders.")


def split_data(root, full_dataname, read_path, tgt_path):
    if (full_dataname == 'cnn') or (full_dataname == 'dailymail'):
        train_urls = root + '/cnn-dailymail/url_lists/{}_wayback_training_urls.txt'.format(full_dataname)
        dev_urls = root + '/cnn-dailymail/url_lists/{}_wayback_validation_urls.txt'.format(full_dataname)
        test_urls = root + '/cnn-dailymail/url_lists/{}_wayback_test_urls.txt'.format(full_dataname)

        move_file_to_dir_url(url_file=train_urls, path_read=read_path,
                             file_to_write=os.path.join(tgt_path, 'train.txt'))

        move_file_to_dir_url(url_file=dev_urls, path_read=read_path,
                             file_to_write=os.path.join(tgt_path, 'dev.txt'))

        move_file_to_dir_url(url_file=test_urls, path_read=read_path,
                             file_to_write=os.path.join(tgt_path, 'test.txt'))
    elif full_dataname == 'nyt':
        # TODO exsisting loading
        raise NotImplementedError
        files = [x.split(".")[0] for x in os.listdir(read_path) if x.endswith(".data")]
        random.shuffle(files)
        total_len = len(files)
        train_len = int(total_len * 0.8)
        dev_len = int(total_len * 0.1)

        move_file_to_dir_file_name(file_list=files[:train_len], path_read=read_path,
                                   file_to_write=os.path.join(tgt_path, 'train.txt'))

        move_file_to_dir_file_name(file_list=files[train_len:train_len + dev_len], path_read=read_path,
                                   file_to_write=os.path.join(tgt_path, 'dev.txt'))

        move_file_to_dir_file_name(file_list=files[train_len + dev_len:], path_read=read_path,
                                   file_to_write=os.path.join(tgt_path, 'test.txt'))

    else:
        raise NotImplementedError


def convert_json_to_pkl_local(root, _data_name):
    convert_json_file_to_pkl_dump(path=root + "/2merge-{}".format(_data_name),
                                  txt_fname="test", part=_data_name)
    print("Test done. Training start.")
    convert_json_file_to_pkl_dump(path=root + "/2merge-{}".format(_data_name),
                                  txt_fname="train", part=_data_name)
    print("Test done. Training done. Dev start.")
    convert_json_file_to_pkl_dump(path=root + "/2merge-{}".format(_data_name),
                                  txt_fname="dev", part=_data_name)


def stage_1():
    clear_dir(path_wt)

    # create_oracles  process_one_example
    # write down data to path_wt
    create_oracles_ilp(dataname=data_name, path_read=path_read, path_wt_distributed=path_wt)

    clear_dir(path_wt_merge)
    print(path_wt_merge)
    # split data
    split_data(root, full_dataname=full_dataname,
               read_path=path_wt,
               tgt_path=path_wt_merge)


from neusum.data.convert_json_to_pkl import set_data_name

if __name__ == '__main__':
    # try
    import sys

    process_one_example("/scratch/cluster/jcxu/data/snlp-output-dm", "", "0000bf554ca24b0c72178403b54c0cca62d9faf8", 40,
                        'dm')
    print(" data_name  servername   stage2only?")
    data_name = sys.argv[1]
    servername = sys.argv[2]
    if servername == 'titan':
        root = '/scratch/cluster/jcxu/data'
    elif servername == 'cc':
        root = '/home/cc'
    elif servername == 'eve':
        root = '/backup3/jcxu/data'
    else:
        raise NotImplementedError
    if len(sys.argv) >= 4:
        only_stage_2 = True
    else:
        only_stage_2 = False
    # data_name = 'nyt'  # or dm or nyt
    print(data_name)
    set_data_name(data_name, servername)

    if data_name == 'cnn':
        full_dataname = 'cnn'
    elif data_name == 'dm':
        full_dataname = 'dailymail'
    else:
        full_dataname = data_name

    # if data_name == 'nyt':
    #     vocab_path = "/home/cc/nyt_vocab"
    # else:
    #     vocab_path = None

    # root = "/home/cc"
    # root = "/backup3/jcxu/data"

    # read the stanford paring output
    path_read = root + "/snlp-output-{}".format(data_name)
    # write spread files
    path_wt = root + "/1oracle-{}".format(data_name)

    path_wt_merge = root + "/2merge-{}".format(data_name)
    if not only_stage_2:
        stage_1()

    # generate vocab ONLY for NYT or only cnn or dm
    # if vocab_path is None:
    #     train_txt = os.path.join(path_wt_merge, 'train.txt')
    #     dev_txt = os.path.join(path_wt_merge, 'dev.txt')
    #     test_txt = os.path.join(path_wt_merge, 'test.txt')
    #     outcome = read_one(train_txt) + read_one(dev_txt) + read_one(test_txt)
    #     vocab = Vocabulary.from_instances(outcome, min_count={'tokens': 30})
    #     vocab.save_to_files(root + "/{}_vocab".format(data_name))
    #     print("saving vocab.")

    # convert json to pkl
    ####
    #### modify vocab_path = "/home/cc/data/cnndm_vocab" in convert json

    ####

    print("converting json/txt to pkl")

    convert_json_to_pkl_local(root, data_name)

    path_transfer = root + "/2merge-{}".format(data_name)
    path_tgt = "jcxu@128.83.143.215:/backup3/jcxu/data/"
    print("scp -r {} {}".format(path_transfer, path_tgt))
    # call("scp -r {} {}".format(path_transfer, path_tgt))
