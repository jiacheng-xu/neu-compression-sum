path = "/scratch/cluster/jcxu/data"
ilp = "2merge-cnndm-ilp/dev.pkl.cnn.00"
beam = "2merge-cnndm/dev.pkl.cnn.00"
import os, json, pickle

if __name__ == '__main__':

    ilp_file = os.path.join(path, ilp)

    f = open(ilp_file, 'rb')
    data = pickle.load(f)
    print("ilp")

    index = [1, 2, 3, 4, 5]
    rouge = []
    # average sentence length
    # average Rouge
    # best rouge
    # rouge delta from last to current: last=A, current=A+x
    for d in data:
        ora = d['_sent_oracle']['4']['data']
        if ora:
            for key, value in ora.items():
                # value['sent']
                pass
        print(ora)
