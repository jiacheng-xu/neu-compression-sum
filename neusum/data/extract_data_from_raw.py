import sys
import os
import hashlib
import struct
import subprocess
import collections
import shutil
from typing import List
import multiprocessing

# path_root = '/home/cc/'
path_root = '/backup3/jcxu/data/'
import os

data_name = 'dm'
if data_name == 'cnn':
    full_dataname = 'cnn'
elif data_name == 'dm':
    full_dataname = 'dailymail'
else:
    raise NotImplementedError

path_data = path_root + 'original-cnndm/{}/stories'.format(full_dataname)

path_data_bf = path_root + 'snlp-{}-parse'.format(data_name)

path_data_af = '/home/cc/snlp-output-{}'.format(data_name)

path_seg = '/home/cc/seg-{}'.format(data_name)

flist = "before-dplp-{}-flist.txt".format(data_name)


# path_final_root = '/home/cc/data-{}'.format(data_name)
# path_final_xml = '/home/cc/data-{}/xml'.format(data_name)
# path_final_merge = '/home/cc/data-{}/merge'.format(data_name)
# path_final_bra = '/home/cc/data-{}/brackets'.format(data_name)


def clear_dir(_path):
    if os.path.isdir(_path):
        shutil.rmtree(_path)
    os.mkdir(_path)


dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
              ")"]  # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


def get_url_hashes(url_list):
    cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cnt)
    out = pool.map(hashhex, url_list)
    pool.close()
    pool.join()
    return out
    # return [hashhex(url) for url in url_list]


def fix_missing_period(line):
    line = line.rstrip()
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line: return line
    if line == "": return line
    if line[-1] in END_TOKENS: return line
    # print line[-1]
    return line + " ."


def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def get_art_abs(story_file):
    lines = read_text_file(story_file)

    # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue  # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article into a single string
    article_lines = article_lines[:30]
    article = '\n'.join(article_lines)

    # Make abstract into a signle string, putting <s> and </s> tags around the sentences
    abstract = '\n'.join([sent for sent in highlights])

    return article, abstract


# Before Stanford NLP:
# Clean files. Filter out blank article or abs.


# get_art_abs(os.path.join(path_dm,'000064fee589e5607c1534a69f852d37b4936cca.story'))

def seperate_file(dir_to_read, name, to_write_dir):
    name_token = name.split('.')[0]
    article, abstract = get_art_abs(os.path.join(dir_to_read, name))
    if len(article) < 30 or len(abstract) < 10:
        print('Discard: {}'.format(name))
        return None
    with open(os.path.join(to_write_dir, name_token + '.doc'), 'w') as fd:
        fd.write(article)
    with open(os.path.join(to_write_dir, name_token + '.abs'), 'w') as fd:
        fd.write(abstract)


def stage1(source_path):
    files = os.listdir(source_path)
    l = len(files)
    print('Start seperating files!')
    cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cnt)
    pool.starmap(seperate_file, zip([source_path] * l, files, [path_data_bf] * l))
    pool.close()
    pool.join()
    print("done")

def run_snlp(i):
    cmd = (
        "java -mx4g -cp /home/cc/stanford-corenlp-full-2018-02-27/* edu.stanford.nlp.pipeline.StanfordCoreNLP "
        "-annotators tokenize,ssplit,pos,lemma,ner,parse "
        " -filelist {} -outputFormat json "
        "-outputDirectory {}".format(
            os.path.join(path_root, data_name + str(i) + flist), path_data_af))
    print(cmd)
    command = cmd.split(' ')
    subprocess.call(command)


def stanford_pre(path_bf):
    files = os.listdir(path_bf)
    files = [os.path.join(path_bf, f) for f in files]

    num_flist = int(multiprocessing.cpu_count())
    slice = len(files) // num_flist
    for i in range(num_flist):
        if i == num_flist - 1:
            tmp = files[i * slice:]
        else:
            tmp = files[i * slice:(i + 1) * slice]
        with open(os.path.join(path_root, data_name + str(i) + flist), 'w') as fd:
            fd.write('\n'.join(tmp))

    # cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_flist)
    pool.map(run_snlp, range(num_flist))
    pool.close()
    pool.join()


# python2 convert.py /home/cc/snlp-output-cnn
# python2 segmenter.py /home/cc/snlp-output-cnn /home/cc/seg-cnn
# python2 rstparser.py /home/cc/seg-cnn False

# dir: path_data_af: all files in xml and conll
# dir: path_seg: all files in brackets and merge
# Store XML merge and brackets

# - three sub folders
#
# - data_dir (XML, merge, brackets)
#     - train
#         - doc
#         - abs
#     - dev
#     - test
split = ['train', 'dev', 'test']
cat = ['doc', 'abs']

# After preprocess, dump data into ...
from shutil import copyfile, move


def _mv_from_to(src_path, fname, tgt_path) -> None:
    if os.path.isfile(os.path.join(src_path, fname)):
        shutil.move(os.path.join(src_path, fname),
                    os.path.join(tgt_path, fname))
    else:
        print("warning: missing file {}".format(fname))


def move(src_path: str
         , tgt_path,
         suffix: str,
         split_name: str,
         url):
    os.mkdir(os.path.join(tgt_path, split_name))  # gen data-cnn/xml/train/
    os.mkdir(os.path.join(tgt_path, split_name, 'doc'))
    os.mkdir(os.path.join(tgt_path, split_name, 'abs'))

    with open(url, 'r', encoding='utf-8') as fd:
        lines = fd.read().splitlines()
        url_names = get_url_hashes(lines)
        print("len of urls {}".format(len(url_names)))
    f_abs = [u + '.abs.' + suffix for u in url_names]
    f_doc = [u + '.doc.' + suffix for u in url_names]

    l = len(f_abs)
    _l = len(f_doc)
    assert l == _l

    cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cnt)
    pool.starmap(_mv_from_to, zip([src_path] * l, f_abs, [os.path.join(tgt_path, split_name, 'abs')] * l))
    pool.close()
    pool.join()

    pool = multiprocessing.Pool(processes=cnt)
    pool.starmap(_mv_from_to, zip([src_path] * l, f_doc, [os.path.join(tgt_path, split_name, 'doc')] * l))
    pool.close()
    pool.join()


if __name__ == '__main__':
    train_urls = '/home/cc/cnn-dailymail/url_lists/{}_wayback_training_urls.txt'.format(full_dataname)
    dev_urls = '/home/cc/cnn-dailymail/url_lists/{}_wayback_validation_urls.txt'.format(full_dataname)
    test_urls = '/home/cc/cnn-dailymail/url_lists/{}_wayback_test_urls.txt'.format(full_dataname)

    # clear_dir(path_final_root)

    clear_dir(path_data_bf)

    stage1(path_data)
    exit()

    clear_dir(path_data_af)
    stanford_pre(path_data_bf)

    exit()

    clear_dir(path_data_af)
    print("python2 /home/cc/DPLP/convert.py {}".format(path_data_af))
    dblp1 = "python2 /home/cc/DPLP/convert.py {}".format(path_data_af)
    dblp1_cmd = dblp1.split(' ')
    subprocess.call(dblp1_cmd)
    clear_dir(path_seg)
    print("python2 /home/cc/DPLP/segmenter.py {} {}".format(path_data_af, path_seg))
    dblp2 = "python2 /home/cc/DPLP/segmenter.py {} {}".format(path_data_af, path_seg)
    dblp2_cmd = dblp2.split(' ')
    subprocess.call(dblp2_cmd)

    print("python2 /home/cc/DPLP/rstparser.py {} False".format(path_seg))
    dblp3 = "python2 /home/cc/DPLP/rstparser.py {} False".format(path_seg)
    dblp3_cmd = dblp3.split(' ')
    subprocess.call(dblp3_cmd)

    clear_dir(path_final_bra)
    clear_dir(path_final_merge)

    move(src_path=path_seg, tgt_path=path_final_bra, suffix='brackets',
         split_name='train', url=train_urls)
    move(src_path=path_seg, tgt_path=path_final_bra, suffix='brackets',
         split_name='dev', url=dev_urls)
    move(src_path=path_seg, tgt_path=path_final_bra, suffix='brackets',
         split_name='test', url=test_urls)

    move(src_path=path_seg, tgt_path=path_final_merge, suffix='merge',
         split_name='train', url=train_urls)
    move(src_path=path_seg, tgt_path=path_final_merge, suffix='merge',
         split_name='dev', url=dev_urls)
    move(src_path=path_seg, tgt_path=path_final_merge, suffix='merge',
         split_name='test', url=test_urls)
