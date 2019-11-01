file = "/backup3/jcxu/visual.txt"
from typing import List


def easy_post_processing(inp: str):
    if len(inp)<=1:
        return ""
    inp = inp.strip()
    if inp[0] in [',', ':', ';', '.', '?', '!']:
        inp = inp[1:]

    inp_str = inp.replace(", ,", "")
    inp_str = inp_str.replace("  ", " ")
    if inp_str[0].islower():
        inp_str = inp_str[0].upper() + inp_str[1:]
    return inp_str


def __remove__(input: str):
    input = input.replace("Visual:", "")
    units = input.split(" ")
    label = []
    for u in units:
        if (u.startswith("_") and u.endswith("_")):
            label.append(True)
        else:
            label.append(False)
    after = []
    for u, l in zip(units, label):
        if not l:
            after.append(u)

    after = " ".join(after)
    sents = after.split("|")
    sents = "|".join([easy_post_processing(s) for s in sents])
    return sents


def test():
    print("test")
    with open(file, 'r') as fd:
        lines = fd.read().splitlines()
    after_lines = [__remove__(l) for l in lines]
    print("\n".join(after_lines))


test()
