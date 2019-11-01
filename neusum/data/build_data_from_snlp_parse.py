# Recover the original files from SNLP parses.
import os, json
from neusum.orac.util import extract_parse, extract_tokens, read_file

SPLIT_SIG = "<SPLIT>"


def recover_one_sample(path_read, fname_without_suffix, data_name, max_sent=30):
    doc_file = os.path.join(path_read, fname_without_suffix + ".doc.json")
    abs_file = os.path.join(path_read, fname_without_suffix + ".abs.json")
    if (not os.path.isfile(doc_file)) or (not os.path.isfile(abs_file)):
        raise TypeError
    try:
        doc_dict = json.loads(read_file(doc_file))
        abs_dict = json.loads(read_file(abs_file))
    except json.decoder.JSONDecodeError:
        print("forget it?")
        return ""
    abs_token, abs_str = extract_tokens(abs_dict)
    doc_token, doc_str = extract_tokens(doc_dict)

    # filter doc
    cnt = 0
    _formal_doc_token = []
    for doc_idx, doc_t in enumerate(doc_token):
        l = len(doc_t)
        if l > 3 and l < 70:  # stat shows l>70+ is  64 out of 26k
            _formal_doc_token.append(doc_t)
            cnt += 1
        if cnt >= max_sent:
            break
    doc_token = _formal_doc_token
    doc_bag, abs_bag = [], []
    for d_token in doc_token:
        d_str = " ".join(d_token)
        doc_bag.append(d_str)
    for a_token in abs_token:
        a_str = " ".join(a_token)
        abs_bag.append(a_str)
    doc_bag = [x for x in doc_bag if x != ""]
    rt_doc = "{}".format(SPLIT_SIG).join(doc_bag)
    abs_bag = [x for x in abs_bag if x != ""]
    rt_abs = "{}".format(SPLIT_SIG).join(abs_bag)
    # cnn-8hu23fh923hf \tdoc_sent_1<SPLIT>doc_sent_2<SPLIT>doc_sent_3<SPLIT>...\tabs_sent_1<SPLIT>abs_sent_2..
    wt = "{}\t{}\t{}\t{}".format(data_name, fname_without_suffix, rt_doc, rt_abs)
    return wt


import multiprocessing

if __name__ == '__main__':
    data_name = "nyt"
    path_to_snlp = "/backup3/jcxu/data/snlp-output-{}".format(data_name)
    wt_out_add = "/backup3/jcxu/data/"
    files = [i.split(".")[0] for i in os.listdir(path_to_snlp) if i.endswith('.doc.json')]
    # files = files[:100]
    total_num = len(files)
    cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cnt)

    out = pool.starmap(recover_one_sample, zip([path_to_snlp] * total_num,
                                               files,
                                               [data_name] * total_num
                                               ))
    pool.close()
    pool.join()
    print("finish creating oracles, and write down them to distribued folders.")

    out = [x for x in out if x != ""]
    print(out[222])
    output = "\n".join(out)
    out_file_name = "sent_{}.txt".format(data_name)
    with open(os.path.join(wt_out_add, out_file_name), 'w') as fd:
        fd.write(output)
    print("Done")
