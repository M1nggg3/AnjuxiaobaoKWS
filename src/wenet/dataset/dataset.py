from __future__ import annotations

from torch.utils.data import IterableDataset

import wekws.dataset.processor as processor
from wekws.dataset.dataset import DataList, Processor
from wekws.utils.file_utils import read_lists


def tokenize(data, tokenizer):
    for sample in data:
        sample["label"] = tokenizer.text2ids(sample["label"])
        yield sample


def tuple_to_dict(data):
    for keys, feats, target, feats_lengths, target_lengths in data:
        yield {
            "keys": keys,
            "feats": feats,
            "target": target,
            "feats_lengths": feats_lengths,
            "target_lengths": target_lengths,
        }


def Dataset(data_type, data_list_file, tokenizer, conf, partition=True):
    if data_type != "raw":
        raise ValueError(f"Only raw data_type is supported for WeKWS compatibility, got {data_type}")

    lists = read_lists(data_list_file)
    shuffle = conf.get("shuffle", True)
    dataset: IterableDataset = DataList(lists, shuffle=shuffle, partition=partition)
    dataset = Processor(dataset, processor.parse_raw)
    filter_conf = conf.get("filter_conf", {})
    dataset = Processor(
        dataset,
        processor.filter,
        max_length=filter_conf.get("max_length", 10240),
        min_length=filter_conf.get("min_length", 10),
    )
    dataset = Processor(dataset, processor.resample, **conf.get("resample_conf", {}))

    if conf.get("speed_perturb", False):
        dataset = Processor(dataset, processor.speed_perturb)

    feats_type = conf.get("feats_type", "fbank")
    if feats_type == "mfcc":
        feature_conf = {"feature_type": "mfcc", **conf.get("mfcc_conf", {})}
        dataset = Processor(dataset, processor.compute_mfcc, **feature_conf)
    elif feats_type == "fbank":
        feature_conf = {"feature_type": "fbank", **conf.get("fbank_conf", {})}
        dataset = Processor(dataset, processor.compute_fbank, **feature_conf)
    else:
        raise ValueError(f"Unsupported feats_type: {feats_type}")

    dataset = Processor(dataset, tokenize, tokenizer)

    if conf.get("spec_aug", True):
        dataset = Processor(dataset, processor.spec_aug, **conf.get("spec_aug_conf", {}))

    if shuffle:
        dataset = Processor(dataset, processor.shuffle, **conf.get("shuffle_conf", {}))

    dataset = Processor(dataset, processor.batch, **conf.get("batch_conf", {}))
    dataset = Processor(dataset, processor.padding)
    dataset = Processor(dataset, tuple_to_dict)
    return dataset
