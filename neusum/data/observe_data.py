
# data_path = "/backup2/jcxu/data/dm/dm-gramTrue-miniFalse-maxsent30-beam8"
# data_path = "/backup2/jcxu/data/cnndm/"
import os
import allennlp


#
# # test.pkl.dm.01
# x= [str(a) for a in x]
# print("\n".join(x))
# exit()
import pickle

def comp_ratio_distribution(data_path, data_name):
    print(data_path)
    files = [x for x in os.listdir(data_path) if x.startswith("test.pkl.{}".format(data_name))]
    bag = []
    for file in files:
        f = open(os.path.join(data_path, file), 'rb')
        data = pickle.load(f)
        for instance_fields in data:
            meta = instance_fields['metadata']
            ratios = instance_fields['comp_rouge_ratio']

            for r in ratios.field_list:
                np_arr = r.array.tolist()
                bag += np_arr
                if len(bag) > 2000:
                    bag = [str(x) for x in bag][:2000]
                    print("\n".join(bag))
                    exit()
            doc_list = meta['doc_list'][0:3]
            abs_list = meta['abs_list']
            # x = [print(len(a)) for a in meta['doc_list'] ]
            doc_list = [" ".join(x) for x in doc_list]
            abs_list = [" ".join(x) for x in abs_list]
if __name__ == '__main__':
    # data_path = "/backup3/jcxu/data/2merge-cnndm"
    # comp_ratio_distribution(data_path, 'dm')
    data_path = "/backup3/jcxu/data/2merge-nyt"
    comp_ratio_distribution(data_path, 'nyt')