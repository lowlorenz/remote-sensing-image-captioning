from dataset import NWPU_Captions
from torchvision.transforms import ToTensor
from model import ImageCaptioningSystem
import pytorch_lightning as pl
import torch
import click
import wandb
from pathlib import Path 

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

def get_data_loaders(bs, train_set, val_set, test_set):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

@click.command()
@click.option('--epochs', default=10, help='Number of epochs to train per cycle.')
@click.option('--maxcycles', default=5, help='Number of active learning cycles to train.')
@click.option('--init_set_size', default=.05, help='Initial train set size in percent.')
@click.option('--new_data_size', default=.05, help='Percentage of added labels per cycle.')
@click.option('--lr', default=.0001, help='Learning rate of the optimizer.')
@click.option('--bs', default=4, help='Batch size.')
@click.option('--sample_method', default='random', help='Sampling method to retrieve more labels.')
@click.option('--device', default='cuda', help='Device to train on.')
@click.option('--run_name', default='test', help='Name of the run.')
@click.option('--data_path', default='NWPU-Captions/', help='Path to the NWPU-Captions dataset.')
@click.option('--debug', is_flag=True, help='Debug mode.')
def train(epochs, maxcycles, init_set_size, new_data_size, lr, bs, sample_method, device, run_name, data_path, debug):
    images = Path(data_path, 'NWPU_images')
    annotations = Path(data_path, 'dataset_nwpu.json')
    train_set = NWPU_Captions(root=images, annotations_file=annotations, split='train', transform=ToTensor())
    val_set = NWPU_Captions(root=images, annotations_file=annotations, split='val', transform=ToTensor())
    test_set = NWPU_Captions(root=images, annotations_file=annotations, split='test', transform=ToTensor())

    train_set.set_empty_mask()
    inital_elements = int(train_set.max_length() * init_set_size)
    train_set.add_random_labels(inital_elements)

    model = ImageCaptioningSystem(lr)
    if debug:
        trainer = pl.Trainer(accelerator=device, devices=1, max_epochs=epochs, limit_train_batches=2, limit_val_batches=2)
    else:
        trainer = pl.Trainer(accelerator=device, devices=1, max_epochs=epochs)

    wandb.init(project="active_learning", config={'epochs': epochs, 'maxcycles': maxcycles, 'init_set_size': init_set_size, 'new_data_size': new_data_size, 'lr': lr, 'bs': bs, 'sample_method': sample_method, 'device': device, 'run_name': run_name}, name=run_name)

    for cycle in range(maxcycles):
        t_loader, v_loader, _ = get_data_loaders(bs, train_set, val_set, test_set)
    
        wandb.log({'cycle': cycle}, step=trainer.current_epoch)
        trainer.fit(model, train_dataloaders=t_loader, val_dataloaders=v_loader)
        if sample_method == 'random':
            elements_to_add = int(train_set.max_length() * new_data_size)
            train_set.add_random_labels(elements_to_add)
        
if __name__ == '__main__':
    train()
