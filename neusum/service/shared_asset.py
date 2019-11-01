global device
global vocab
global device_id


def get_device():
    return device


def get_device_id():
    return device_id


def set_device_id(x):
    global device_id
    device_id = x


def set_device(x):
    global device
    device = x


def get_vocab():
    return vocab


def set_vocab(x):
    global vocab
    vocab = x
