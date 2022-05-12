from license_plate_detection_recognition.lprnet.model.lprnet import LPRNet
from license_plate_detection_recognition.lprnet.model.stn import STNet
from license_plate_detection_recognition.lprnet.data.cropped_lps_dataset import LPRDataset, collate_fn, CHARS
import license_plate_detection_recognition.lprnet.decoders as d
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import torchvision
import matplotlib.pyplot as plt


def convert_image(inp):
    # convert a Tensor to numpy image
    inp = inp.numpy().transpose((1, 2, 0))
    inp = 127.5 + inp/0.0078125
    inp = inp.astype('uint8') 
    inp = inp[:, :, ::-1]
    return inp


if __name__ == '__main__':
    batch_size = 1
    parser = argparse.ArgumentParser(description='LPR view on dataset')
    parser.add_argument('--img_size', default=(94, 24), help='the image size')
    parser.add_argument('--img_dirs', default="D:/Share/test", help='the images path')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)
    lprnet.to(device)
    # lprnet.load_state_dict(torch.load('weights/Final_LPRNet_model.pth', map_location=lambda storage, loc: storage))
    checkpoint = torch.load('saving_ckpt/lprnet_Iter_678900_model.ckpt')
    lprnet.load_state_dict(checkpoint['net_state_dict'])
    lprnet.eval() 
    print("LPRNet loaded")
    
#    torch.save(lprnet.state_dict(), 'weights/Final_LPRNet_model.pth')
    
    STN = STNet()
    STN.to(device)
    # STN.load_state_dict(torch.load('weights/STN_model_Init.pth', map_location=lambda storage, loc: storage))
    STN.load_state_dict(torch.load('weights/Final_STN_model.pth', map_location=lambda storage, loc: storage))
    #checkpoint = torch.load('saving_ckpt/stn_Iter_678900_model.ckpt')
    #STN.load_state_dict(checkpoint['net_state_dict'])
    STN.eval()
    print("STN loaded")
    
#    torch.save(STN.state_dict(), 'weights/Final_STN_model.pth')
    
    dataset = LPRDataset([args.img_dirs], args.img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
    print('dataset loaded with length : {}'.format(len(dataset)))

    with torch.no_grad():
        for imgs, labels, lengths in dataloader:  # img: torch.Size([2, 3, 24, 94])  # labels: torch.Size([14]) # lengths: [7, 7] (list)
            imgs, labels = imgs.to(device), labels.to(device)
            transfer = STN(imgs)
            # transfer = imgs
            logits = lprnet(transfer)  # torch.Size([batch_size, CHARS length, output length ])

            preds = logits.cpu().detach().numpy()  # (batch size, 68, 18)
            pred_chars, pred_labels = d.decode_greedy(preds, CHARS)  # list of predict output

            lb = ""
            for i in labels:
                lb += CHARS[int(i)]

            input_tensor = imgs.cpu()
            transformed_input_tensor = transfer.cpu()

            in_grid = convert_image(torchvision.utils.make_grid(input_tensor))
            out_grid = convert_image(torchvision.utils.make_grid(transformed_input_tensor))

            # Plot the results side-by-side
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(in_grid)
            axarr[0].set_title(f'Dataset Image {lb}')

            axarr[1].imshow(out_grid)
            axarr[1].set_title(f'Transformed Image {pred_chars[0]}')
            plt.waitforbuttonpress()
            plt.close(f)
            continue



