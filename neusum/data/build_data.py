from allennlp.data.vocabulary import Vocabulary
from neusum.data.cnndm_dataset_read_pkl import SummarizationDatasetReaderPkl
import os

import logging


def data_builder(dbg: bool = False,
                 lazy=True,
                 dataset_dir: str = "/backup2/jcxu/data/cnn-v1/read-ready-files",
                 train_file: str = "train.pkl",
                 dev_file: str = "dev.pkl",
                 test_file: str = 'test.pkl',
                 single_oracle=True,
                 fix_edu_num=None,
                 trim_sent_oracle: bool = True,
                 save_to: str = None):
    reader = SummarizationDatasetReaderPkl(lazy=lazy, dir=dataset_dir,
                                           single_oracle=single_oracle, fix_edu_num=fix_edu_num,
                                           trim_sent_oracle=trim_sent_oracle,
                                           vocab_path=os.path.join(dataset_dir, dev_file),
                                           save_to=save_to,dbg=dbg
                                           )

    if dbg:
        # dev_file = "minidev.txt"
        dev_data = reader.read(dev_file)

        test_data = reader.read(test_file)
        train_data = dev_data
    else:
        train_data = reader.read(train_file)
        test_data = reader.read(test_file)
        dev_data = reader.read(dev_file)

    # vocab = Vocabulary.from_instances(instances=train_data + dev_data + test_data,
    #                                   min_count={"tokens": 5})
    # logging.info("Vocab size: {}".format(int(vocab.get_vocab_size())))
    return train_data, dev_data, test_data, reader.vocab, reader.word_token_indexers
