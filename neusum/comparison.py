from neusum.evaluation.rouge_with_pythonrouge import RougeStrEvaluation
import os

r = RougeStrEvaluation(name="test")


def read_story(file):
    with open(file, 'r') as fd:
        liens = fd.read().splitlines()
    d = {}
    for l in liens:
        part, uid, doc, abs = l.split("\t")
        abs_list = abs.split("<SPLIT>")
        d[uid] = abs_list
    return d


def test_xxz():
    test = "dm"

    root = "/backup3/jcxu/data"

    xxz = root + "/xxz-latent/xxz-output"

    if test == 'cnndm':
        files = [x for x in os.listdir(xxz)]
        d_abs_cnn = read_story(os.path.join(root, "sent_cnn.txt"))
        d_abs_dm = read_story(os.path.join(root, "sent_dm.txt"))
        d_abs = {**d_abs_cnn, **d_abs_dm}
    else:

        data_name = test
        fname = "sent_{}.txt".format(data_name)
        files = [x for x in os.listdir(xxz) if x.startswith(data_name)]
        d_abs = read_story(os.path.join(root, fname))

    for f in files:
        part, remain = f.split("-")
        uid = remain.split(".")[0]
        abs = d_abs[uid]

        with open(os.path.join(xxz, f), 'r') as fd:
            p = fd.read().splitlines()
            r(pred=p, ref=[abs])
    print(len(r.pred_str_bag))
    r.get_metric(reset=True)


if __name__ == '__main__':
    test_xxz()
