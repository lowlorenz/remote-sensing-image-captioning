from dataset import NWPU_Captions
from torchvision.transforms import ToTensor
from model import ImageCaptioningSystem
import pytorch_lightning as pl
import torch

hparams = {
    'device': 'cuda',
    'sample_method': 'random',
    'bs': 4,
    'lr': 0.0001,
    'epochs': 1,
    'epochs_total': 10,
    'maxcycles': 5,
    'init_set_size': .05,
    'new_data_size': .05,
}

def get_data_loaders(hparams):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=hparams['bs'], shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=hparams['bs'], shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=hparams['bs'], shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':

    train_set = NWPU_Captions(root='../NWPU_images', annotations_file='../dataset_nwpu.json', split='train', transform=ToTensor())
    val_set = NWPU_Captions(root='../NWPU_images', annotations_file='../dataset_nwpu.json', split='val', transform=ToTensor())
    test_set = NWPU_Captions(root='../NWPU_images', annotations_file='../dataset_nwpu.json', split='test', transform=ToTensor())

    train_set.set_empty_mask()
    train_set.random_labeling(int(train_set.max_length() * hparams['init_set_size']))

    model = ImageCaptioningSystem(hparams['lr'])

    t_loader, v_loader, _ = get_data_loaders(hparams)

    trainer = pl.Trainer(accelerator='cuda', devices=1, max_epochs=100, limit_train_batches=0.1, limit_val_batches=0.1)
    trainer.fit(model, train_dataloaders=t_loader, val_dataloaders=v_loader)