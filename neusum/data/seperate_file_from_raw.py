import os
import hashlib
import multiprocessing
import shutil


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
    article_lines = article_lines[:35]
    article = '\n'.join(article_lines)

    # Make abstract into a signle string, putting <s> and </s> tags around the sentences
    abstract = '\n'.join([sent for sent in highlights])

    return article, abstract


def seperate_file(dir_to_read, name, to_write_dir):
    name_token = name.split('.')[0]
    article, abstract = get_art_abs(os.path.join(dir_to_read, name))
    if len(article) < 5 or len(abstract) < 5:
        print('Discard: {}'.format(name))
        return None
    with open(os.path.join(to_write_dir, name_token + '.doc'), 'w') as fd:
        fd.write(article)
    with open(os.path.join(to_write_dir, name_token + '.abs'), 'w') as fd:
        fd.write(abstract)


def stage1(source_path, _path_data_bf):
    files = os.listdir(source_path)
    print('Start seperating files!')
    l = len(files)
    cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cnt)
    pool.starmap(seperate_file, zip([source_path] * l, files, [_path_data_bf] * l))
    pool.close()
    pool.join()


import sys

if __name__ == '__main__':
    data_name, path_root, path_seped = sys.argv[1:]
    print("Preprocessing from raw xx/stories")
    # data_name = 'dailymail'
    # path_root = '/home/cc'
    # path_seped = '/home/cc/seped'

    path_to_raw_data = os.path.join(path_root, data_name, 'stories')
    print('Raw data: {}'.format(path_to_raw_data))

    real_path_seped = path_seped + data_name
    clear_dir(real_path_seped)
    stage1(source_path=path_to_raw_data, _path_data_bf=real_path_seped)
    print("Seperated files in {}".format(real_path_seped))
