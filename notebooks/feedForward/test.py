import CatalinaLCDataSet as ds
import torchvision
import torch

transformed_dataset = ds.dataSet('train',
                                 transform=torchvision.transforms.Compose([
                                     ds.ToTensor()
                                 ]))


dataloader = torch.utils.data.DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True)



for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['features'].size(),
          sample_batched['label'].size())

    print(sample_batched['features'])
    print(sample_batched['label'])
    # observe 4th batch and stop.
    if i_batch == 3:
        break