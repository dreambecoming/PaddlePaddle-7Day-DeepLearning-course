import jieba
import numpy as np
import paddle


def convert_example(example, vocab, unk_token_id=1, is_test=False):
    """
    jieba 分词，转换id
    """

    input_ids = []
    for token in jieba.cut(example[0]):
        token_id = vocab.get(token, unk_token_id)
        input_ids.append(token_id)
    valid_length = np.array(len(input_ids), dtype='int64')

    if not is_test:
        label = np.array(example[-1], dtype="int64")
        return input_ids, valid_length, label
    else:
        return input_ids, valid_length


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n").split("\t")[0]
        vocab[token] = index
    return vocab


def convert_tokens_to_ids(tokens, vocab):
    """ Converts a token id (or a sequence of id) in a token string
        (or a sequence of tokens), using the vocabulary.
    """

    ids = []
    unk_id = vocab.get('[UNK]', None)
    for token in tokens:
        wid = vocab.get(token, unk_id)
        if wid:
            ids.append(wid)
    return ids


def create_dataloader(dataset,
                      trans_fn=None,
                      mode='train',
                      batch_size=1,
                      use_gpu=False,
                      pad_token_id=0,
                      batchify_fn=None):
    """
    Creats dataloader.

    Args:
        dataset(obj:`paddle.io.Dataset`): Dataset instance.
        trans_fn(obj:`callable`, optional, defaults to `None`): function to convert a data sample to input ids, etc.
        mode(obj:`str`, optional, defaults to obj:`train`): If mode is 'train', it will shuffle the dataset randomly.
        batch_size(obj:`int`, optional, defaults to 1): The sample number of a mini-batch.
        use_gpu(obj:`bool`, optional, defaults to obj:`False`): Whether to use gpu to run.
        pad_token_id(obj:`int`, optional, defaults to 0): The pad token index.
        batchify_fn(obj:`callable`, optional, defaults to `None`): function to generate mini-batch data by merging
            the sample list, None for only stack each fields of sample in axis
            0(same as :attr::`np.stack(..., axis=0)`).

    Returns:
        dataloader(obj:`paddle.io.DataLoader`): The dataloader which generates batches.
    """
    if trans_fn:
        dataset = dataset.apply(trans_fn, lazy=True)

    if mode == 'train' and use_gpu:
        sampler = paddle.io.DistributedBatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=True)
    else:
        shuffle = True if mode == 'train' else False
        sampler = paddle.io.BatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    dataloader = paddle.io.DataLoader(
        dataset,
        batch_sampler=sampler,
        return_list=True,
        collate_fn=batchify_fn)
    return dataloader
