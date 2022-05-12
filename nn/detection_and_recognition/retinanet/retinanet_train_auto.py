from commons.custom_iterator.generator_init import create_generators
import torch
import torchvision
import torch.optim as optim
import os
from commons.preprocessing import retina_preprocess


batch_size = 16
epoch_to_start = 7  # epoch to start from
epochs = 100


num_classes = 1
dataset_type = 'pascal_custom'
dataset_path='E:\\Datasets\\LicensePlate\\PhotoBaseFull'
custom_classes = {
    'plate number': 0,
}


model_path = f'./{dataset_type}_weights/retinanet.pth'


def main():
    # create the generators
    train_generator, validation_generator = create_generators(dataset_type=dataset_type, dataset_path=dataset_path, batch_size=batch_size, custom_classes=custom_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained_backbone=True, num_classes=num_classes)
    model.train()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    checkpoint_path = f'./{dataset_type}_weights/retinanet_{epoch_to_start}.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])  #, strict=False
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'weights {checkpoint_path} has been loaded')

    batchs_per_epoch = train_generator.size() // batch_size

    for epoch in range(epoch_to_start, epochs):
        batches = 0
        for sample in train_generator:
            batches += 1
            print(f'batch # {batches} of {batchs_per_epoch}')
            image_group, annotations_groups = sample

            # temporary workaround for images without annotations
            skip_batch = False
            for annotations in annotations_groups:
                if len(annotations['labels']) == 0:
                    skip_batch = True
            if skip_batch:
                continue

            image_group_on_device = []
            annotations_group_on_device = []
            for i in range(batch_size):
                image = retina_preprocess(image_group[i])

                annotations = annotations_groups[i]
                image_group_on_device.append(torch.tensor(image, dtype=torch.float).to(device))
                annotations_group_on_device.append({'boxes': torch.tensor(annotations['boxes']).to(device), 'labels': torch.tensor(annotations['labels'], dtype=torch.long).to(device)})

            loss = model(image_group_on_device, annotations_group_on_device)
            cls_logits = loss['classification']
            bbox_regression = loss['bbox_regression']
            total_loss = cls_logits.sum() + bbox_regression.sum()
            print(f'cls_logits: {cls_logits} bbox_regression: {bbox_regression} total_loss: {total_loss}')
            total_loss.backward()
            optimizer.step()

            if total_loss < 0.2:
                torch.save(model.state_dict(), model_path)
                print('Finished Training (early stop)')
                exit()

            if batches >= batchs_per_epoch:
                # we need to break the loop by hand because the generator loops indefinitely
                break

        checkpoint_path = f'./{dataset_type}_weights/retinanet_{epoch + 1}.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

    torch.save(model.state_dict(), model_path)
    print('Finished Training')


if __name__ == '__main__':
    main()
