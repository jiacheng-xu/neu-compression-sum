import os
import shutil
import subprocess
import multiprocessing


def run_snlp( inp):
    filelist, output_dir = inp
    cmd = (
        "java -mx4g -cp /home/cc/stanford-corenlp-full-2018-02-27/* edu.stanford.nlp.pipeline.StanfordCoreNLP "
        "-annotators tokenize,ssplit,pos,lemma,ner,parse "
        " -filelist {} -outputFormat json  "
        "-outputDirectory {}".format(
            filelist, output_dir))
    # cmd = (
    #     "java -mx4g -cp /home/jcxu/stanford-corenlp-full-2018-02-27/* edu.stanford.nlp.pipeline.StanfordCoreNLP "
    #     "-annotators tokenize,ssplit,pos,lemma,ner,parse "
    #     " -filelist {} -outputFormat json  "
    #     "-outputDirectory {}".format(
    #         os.path.join(path_root, data_name + str(i) + flist), path_data_af))

    print(cmd)
    command = cmd.split(' ')
    subprocess.call(command)


def stanford_pre(path_bf,
                 path_af,
                 path_root='/home/cc',
                 data_name='nyt',flist='before-parse-problems-flist.txt'):
    files = os.listdir(path_bf)
    files = [os.path.join(path_bf, f) for f in files]

    num_flist = multiprocessing.cpu_count()
    slice = len(files) // num_flist
    for i in range(num_flist):
        if i == num_flist - 1:
            tmp = files[i * slice:]
        else:
            tmp = files[i * slice:(i + 1) * slice]
        with open(os.path.join(path_root, data_name + str(i) + flist), 'w') as fd:
            fd.write('\n'.join(tmp))

    file_lists = [os.path.join(path_root, data_name + str(i) + flist) for i in range(num_flist)]
    inp = zip(file_lists, [path_af]*len(file_lists))
    # cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_flist)
    pool.map(run_snlp, inp)
    pool.close()
    pool.join()


def clear_dir(_path):
    if os.path.isdir(_path):
        shutil.rmtree(_path)
    os.mkdir(_path)


if __name__ == '__main__':

    data_name = 'nyt'

    # path_root = '/backup3/jcxu/data/'
    path_root = '/home/cc/'

    flist = "before-dplp-problems-flist.txt"

    path_data_bf = path_root + 'snlp-{}-parse'.format(data_name)
    # name.doc

    path_data_af = path_root + 'snlp-output-{}'.format(data_name)
    # name.doc.xml

    path_problems = path_root + 'snlp-problem-{}'.format(data_name)

    clear_dir(path_problems)
    files = os.listdir(path_data_bf)
    cnt = 0
    for f in files:
        if os.path.isfile(os.path.join(path_data_af, f + '.json')):
            continue
        else:
            cnt += 1
            shutil.copyfile(os.path.join(path_data_bf, f), os.path.join(path_problems, f))

    print("Missing {} files".format(cnt))

    stanford_pre(path_problems,
                 path_af=path_data_af)