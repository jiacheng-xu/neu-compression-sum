# Compute Lead3 Baseline
import time

from pythonrouge.pythonrouge import Pythonrouge


def get_best_rouge():
    pass


def oracle_baseline(path_file):
    with open(path_file, 'r') as fd:
        lines = fd.read().splitlines()
    pred_str_bag, ref_str_bag = [], []
    for l in lines:
        name, doc, abst, span_info, gold = l.split('\t')
        doc = doc.split()
        span_info = [int(w) for w in span_info.split()]
        idx_in_span = list(zip(span_info[0::2], span_info[1::2]))
        gold_label = [int(l) for l in gold.split()]

        abs_str = abst.replace("@@SS@@", "\n").split("\n")
        abs_str = [x for x in abs_str if len(x) > 1]
        _buff = []
        for g in gold_label:
            content = doc[idx_in_span[g][0]:idx_in_span[g][1] + 1]
            _buff.append(' '.join(content).replace("@@SS@@", ""))

        pred_str_bag.append(_buff)
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


oracle_baseline("/backup2/jcxu/data/cnn-v1/read-ready-files/sent-dev-100-beam.txt")
print("test oracle")
oracle_baseline("/backup2/jcxu/data/cnn-v1/read-ready-files/test.txt")
