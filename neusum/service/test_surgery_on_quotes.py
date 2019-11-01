from unittest import TestCase

from neusum.service.basic_service import surgery_on_quotes,meta_str_surgery


class TestSurgery_on_quotes(TestCase):
    def test_surgery_on_quotes(self):
        inp_str = "asd , ''hghghghghghghg"
        out = meta_str_surgery(inp_str)
        print("---"+out)
        inp_str = "asd . ''hghghghghghghg"
        out = meta_str_surgery(inp_str)
        print("---"+out)

if __name__ == '__main__':
    t = TestSurgery_on_quotes()
    t.test_surgery_on_quotes()