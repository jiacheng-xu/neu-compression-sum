from unittest import TestCase

from neusum.service.basic_service import easy_post_processing

class TestEasy_post_processing(TestCase):
    def test_easy_post_processing(self):
        inp=[
            "In two years ' time , the Scandinavian nation is slated to become the first in the world to phase out radio entirely .",
            "Digitally , there are four times that number .",
            "Frum : Ukrainians want to enter EU and lessen dependence on Russia ; Putin fighting to stop it .",
            "-LRB- CNN -RRB- He might have just won one of sport 's most prestigious events , but it was n't long before Jordan Spieth 's thoughts turned to his autistic sister in the glow of victory . "
        ]
        for x in inp:
            y=easy_post_processing(x)
            print(y)