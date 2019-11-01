# Compute Lead3 Baseline
import time

from pythonrouge.pythonrouge import Pythonrouge


def lead3_baseline(path_file):
    with open(path_file, 'r') as fd:
        lines = fd.read().splitlines()
    pred_str_bag, ref_str_bag = [], []
    for l in lines:
        name, doc, abst, span_info, gold = l.split('\t')
        doc = doc.split()
        indices = [i for i, x in enumerate(doc) if x == "@@SS@@"]
        abs_str = abst.replace("@@SS@@", "\n").split("\n")
        abs_str = [x for x in abs_str if len(x) > 1]
        if len(indices) > 2:
            sent1 = ' '.join(doc[:indices[0]]).replace("@@SS@@", "")
            sent2 = ' '.join(doc[indices[0]:indices[1]]).replace("@@SS@@", "")
            sent3 = ' '.join(doc[indices[1]:indices[2]]).replace("@@SS@@", "")
            lead3 = [sent1, sent2, sent3]
        else:
            lead3 = ' '.join(doc).replace("@@SS@@", "\n").split("\n")
            lead3 = [x for x in lead3 if len(x) > 1]
        pred_str_bag.append(lead3)
        ref_str_bag.append([abs_str])
    print('Finish reading')
    rouge = Pythonrouge(summary_file_exist=False,
                        summary=pred_str_bag, reference=ref_str_bag,
                        n_gram=2, ROUGE_SU4=True, ROUGE_L=True, ROUGE_W=True,
                        ROUGE_W_Weight=1.2,
                        recall_only=False, stemming=True, stopwords=False,
                        word_level=True, length_limit=False, length=50,
                        use_cf=False, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5, default_conf=True)
    score = rouge.calc_score()
    print(score)


lead3_baseline("/backup2/jcxu/data/cnn-v1/read-ready-files/test.txt")
print("test lead")
lead3_baseline("/backup2/jcxu/data/cnn-v1/read-ready-files/test.txt")
