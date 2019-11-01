import csv
from collections import Counter


def read_a_problem(c, ans_index, field, p_name='p1'):
    ans = decode_ans(c[ans_index], p_name)
    # l1, l2, l3 = c[f1_index[1]], c[f1_index[3]], c[f1_index[5]]
    labels = [c[field[1]], c[field[3]], c[field[5]]]
    d = Counter(["{}{}".format(0, labels[ans[0]]),
                 "{}{}".format(1, labels[ans[1]]),
                 "{}{}".format(2, labels[ans[2]])])
    # d.update("{}{}".format(ans[0],labels[0]))
    # d.update("{}{}".format(ans[1], labels[1]))
    # d.update("{}{}".format(ans[2], labels[2]))
    return d


import json


def decode_ans(ans_json, px='p1'):
    ans = json.loads(ans_json)[0]
    ans1 = [ans[key]['on'] for key in [px + '1a', px + '1b', px + '1c']]
    if any(ans1):
        x = ans1.index(True)
    else:
        x = -1

    ans2 = [ans[key]['on'] for key in [px + '2a', px + '2b', px + '2c']]
    if any(ans2):
        y = ans2.index(True)
    else:
        y = -1

    if x == -1 or y == -1 or x == y:
        return [0, 0, 0]
    else:
        l = list(range(3))
        l.remove(x)
        l.remove(y)
        z = l[0]
        return [x, y, z]


if __name__ == '__main__':
    with open("Batch_3649624_batch_results.csv", 'r', newline='', encoding='utf-8') as fd:
        reader = csv.reader(fd, delimiter=',')
        lines = []
        for row in reader:
            lines.append(row)
        cnt = Counter()
        names = lines[0]
        content = lines[1:]

        fields = ['sent_1', 'label_1', 'sent_2', 'label_2', 'sent_3', 'label_3']
        f1 = ['Input.p1_{}'.format(x) for x in fields]
        f2 = ['Input.p2_{}'.format(x) for x in fields]
        f3 = ['Input.p3_{}'.format(x) for x in fields]
        f4 = ['Input.p4_{}'.format(x) for x in fields]

        ans = 'Answer.taskAnswers'
        ans_index = names.index(ans)
        f1_index = [names.index(x) for x in f1]
        f2_index = [names.index(x) for x in f2]
        f3_index = [names.index(x) for x in f3]
        f4_index = [names.index(x) for x in f4]
        meta_counter = Counter()
        for c in content:
            cnt1 = read_a_problem(c,
                                  ans_index, f1_index, 'p1')
            # ans1 = decode_ans(c[ans_index], 'p1')
            # l1, l2, l3 = c[f1_index[1]], c[f1_index[3]], c[f1_index[5]]
            # labels = [c[f1_index[1]], c[f1_index[3]], c[f1_index[5]]]
            cnt2 = read_a_problem(c, ans_index, f2_index, 'p2')
            cnt3 = read_a_problem(c, ans_index, f3_index, 'p3')
            cnt4 = read_a_problem(c, ans_index, f4_index, 'p4')
            cnt = cnt1 + cnt2 + cnt3 + cnt4
            meta_counter = meta_counter + cnt
        print(meta_counter)
