import json, os, multiprocessing, random
import hashlib,sys
from neusum.orac.util import comp_document_oracle
from neusum.orac.oracle import read_file_no_splitlines, process_one_sentence
from neusum.data.create_oracle import extract_parse, extract_tokens


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


def get_url_hashes(url_list):
    cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cnt)
    out = pool.map(hashhex, url_list)
    pool.close()
    pool.join()
    return out


def read_one_file(fname):
    if os.path.isfile(fname + '.data'):
        with open(fname + '.data', 'r') as fd:
            line = fd.read().splitlines()
            assert len(line) == 1
        return line[0]
    else:
        return None


def move_file_to_dir_url(url_file, path_read, file_to_write):
    with open(url_file, 'r', encoding='utf-8') as fd:
        lines = fd.read().splitlines()
        url_names = get_url_hashes(lines)
        print("len of urls {}".format(len(url_names)))

    url_names = [os.path.join(path_read, url) for url in url_names]
    cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cnt)
    rt_bag = pool.map(read_one_file, url_names)

    pool.close()
    pool.join()

    rt_bag = [x for x in rt_bag if x is not None]
    wt_string = "\n".join(rt_bag)
    with open(file_to_write, 'w') as fd:
        fd.write(wt_string)


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


def stage_1():
    clear_dir(path_wt)

    # create_oracles  process_one_example
    # write down data to path_wt
    wrap_creat_distributed_oracle(dataname=data_name, path_read=path_read, path_wt_distributed=path_wt)

    clear_dir(path_wt_merge)
    print(path_wt_merge)
    # split data
    split_data(root, full_dataname=full_dataname,
               read_path=path_wt,
               tgt_path=path_wt_merge)


def wrap_creat_distributed_oracle(dataname, path_read, path_wt_distributed):
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
                                          [30] * total_num,
                                          [dataname] * total_num))

    # pool.starmap(process_one_example_allen, zip([path_read] * total_num,
    #                                             [path_wt_distributed] * total_num,
    #                                             files,
    #                                             [35] * total_num,
    #                                             [dataname] * total_num))
    pool.close()
    pool.join()
    print("finish creating oracles, and write down them to distributed folders.")


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
    sent_ora_json = comp_document_oracle(doc_list, abs_str)

    # comp document level delete-one sentence oracle
    doc_list_trimmed_for_oracle = []
    for x in sent_packs:
        _del_span = x['del_span']
        _del_span.sort(key=lambda y: y['rouge'], reverse=True)
        try:
            del_idx = _del_span[0]['selected_idx']
            selected_set = list(set(list(range(len(x['token'])))) - set(del_idx))
            selected_set.sort()
        except IndexError:
            selected_set = list(range(len(x['token'])))
        doc_list_trimmed_for_oracle.append(" ".join([x['token'][kdx] for kdx in selected_set]))
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

    # rt["sent_oracle"] = trim_ora_json
    rt["sent_oracle"] = sent_ora_json
    json_rt = json.dumps(rt)
    with open(os.path.join(path_write, fname_without_suffix + '.data'), 'w') as fd:
        fd.write(json_rt)
if __name__ == '__main__':

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
    path_wt = root + "/1oracle-{}-exp".format(data_name)

    path_wt_merge = root + "/2merge-{}-exp".format(data_name)
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
    from subprocess import call

    path_transfer = root + "/2merge-{}".format(data_name)
    path_tgt = "jcxu@128.83.143.215:/backup3/jcxu/data/"
    # call("scp -r {} {}".format(path_transfer, path_tgt))

    # random.shuffle(files)
    # files = files[:50]
    # global_dict = {}
    # for f in files:
    #     stat = process_one_example(path_read, path_wt, f)
    #     if "top9" in stat:
    #         for key, value in stat.items():
    #             if key in global_dict.keys():
    #                 global_dict[key] = global_dict[key] + [value]
    #             else:
    #                 global_dict[key] = [value]
    # # print(global_dict)
    # for key, value in global_dict.items():
    #     value = [str(v) for v in value]
    #     pp = "\t".join(value)
    #     print("{}\t{}".format(key, pp))
