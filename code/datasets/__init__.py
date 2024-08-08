from header import *
from .samplers import DistributedBatchSampler
import torchvision.transforms as transforms
from datasets.randaugment import RandomAugment
from datasets.deepfake_dataset import DGM4_Dataset
from PIL import Image


def create_dataset(config, is_train = True):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    train_transform = transforms.Compose([
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    if is_train:
        DGM4dataset = DGM4_Dataset(config=config, ann_file=config['train_file'], transform=train_transform,
                                   max_words=config['max_words'], is_train=True)
    else:
        DGM4dataset = DGM4_Dataset(config=config, ann_file=config['val_file'], transform=test_transform,
                                   max_words=config['max_words'], is_train=False)
    return DGM4dataset

def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders

def load_DGM4_dataset(dataset, args):
    '''
    tokenizer = get_tokenizer(args['model_path'])
    dataset_name = args['models'][args['model']]['stage1_train_dataset'] # SupervisedDataset, str
    data_path = args["data_path"]
    data = globals()[dataset_name](data_path, tokenizer, args['max_length']) #SupervisedDataset
    '''
    #data = SupervisedDataset(args['data_path'], args['image_root_path'])

    sampler = torch.utils.data.RandomSampler(dataset)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    batch_sampler = DistributedBatchSampler(
        sampler,
        batch_size,
        True,
        rank,
        world_size
    )
    iter_ = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=8,
        collate_fn=dataset.collate,
        pin_memory=False
    )
    return iter_, sampler