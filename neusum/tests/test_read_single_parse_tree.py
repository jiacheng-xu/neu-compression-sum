from unittest import TestCase
from neusum.data.create_oracle import read_single_parse_tree, find_deletable_span_rule_based


class TestRead_single_parse_tree(TestCase):
    def test_read_single_parse_tree(self):
        s = "(ROOT\n  (NP\n    (NP (NN NEW))\n    (: :)\n    (NP (NN Diagnosis))\n    (: :) (`` ``)\n    (S\n      (NP\n        (NP (NN autism))\n        (, ,)\n        (NP\n          (NP (JJ severe) (NN anxiety))\n          (, ,)\n          (NP (JJ post-traumatic) (NN stress) (NN disorder)\n            (CC and)\n            (NN depression) ('' '') (NNP Burkhart))))\n      (VP (VBZ is)\n        (ADVP (RB also))\n        (VP (VBN suspected)\n          (PP (IN in)\n            (NP (DT a) (JJ German) (NN arson) (NN probe))))))\n    (, ,)\n    (S\n      (NP (NNS officials))\n      (VP (VBP say)))\n    (. .)))"
        out = read_single_parse_tree(s)
        rules = ["PP", "SBAR", "ADVP", "ADJP", "S"]
        print(out)
        find_deletable_span_rule_based(rules, out)
