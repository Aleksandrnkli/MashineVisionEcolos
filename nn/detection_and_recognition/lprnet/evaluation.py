from model.lprnet import LPRNet
from model.stn import STNet
import decoders as d
from data.cropped_lps_dataset import LPRDataset, collate_fn, CHARS_ru, CHARS_kz
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
    inp = inp[:, :, ::-1]  # TODO check neccessity of this as there is no corresponding reverse in lprnet_preprocess()
    return inp


def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        dataset = LPRDataset([args.img_dirs], args.img_size)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn) 
        imgs, labels, lengths = next(iter(dataloader))
        
        input_tensor = imgs.cpu()
        transformed_input_tensor = STN(imgs.to(device)).cpu()
        
        in_grid = convert_image(torchvision.utils.make_grid(input_tensor))
        out_grid = convert_image(torchvision.utils.make_grid(transformed_input_tensor))
        
        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')
        
        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')
        plt.waitforbuttonpress()


def eval(class_name, lprnet, STN, dataloader, dataset, device):
    if class_name == 'RU':

        CHARS = CHARS_ru
    elif class_name == 'KZ':
        CHARS = CHARS_kz

    lprnet = lprnet.to(device)
    STN = STN.to(device)
    TP = 0
    for imgs, labels, lengths in dataloader:   # img: torch.Size([2, 3, 24, 94])  # labels: torch.Size([14]) # lengths: [7, 7] (list)
        imgs, labels = imgs.to(device), labels.to(device)
        transfer = STN(imgs)
        logits = lprnet(transfer)  # torch.Size([batch_size, CHARS length, output length ])
    
        preds = logits.cpu().detach().numpy()  # (batch size, 68, 18)
        _, pred_labels = d.beam_search_decoder(class_name, preds, CHARS)  # list of predict output

        start = 0
        for i, length in enumerate(lengths):
            label = labels[start:start+length]
            start += length
            if np.array_equal(np.array(pred_labels[i]), label.cpu().numpy()):
                TP += 1
            
    ACC = TP / len(dataset) 
    
    return ACC
    

if __name__ == '__main__':
    CLASS = 'RU' #for russian lprs
    #CLASS = 'KZ' # for kx lprs
    if CLASS == 'RU':
        CHARS = CHARS_ru
    elif CLASS == 'KZ':
        CHARS = CHARS_kz

    parser = argparse.ArgumentParser(description='LPR Evaluation')
    parser.add_argument('--img_size', default=(94, 24), help='the image size')
    parser.add_argument('--img_dirs', default=f"./Cropped/{CLASS}", help='the images path')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--batch_size', default=128, help='batch size.')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=args.dropout_rate)
    lprnet.to(device)
    # lprnet.load_state_dict(torch.load('weights/Final_LPRNet_model.pth', map_location=lambda storage, loc: storage))
    checkpoint = torch.load('saving_ckpt/lprnet_Iter_000900_model.ckpt')
    lprnet.load_state_dict(checkpoint['net_state_dict'])
    lprnet.eval() 
    print("LPRNet loaded")
    
#    torch.save(lprnet.state_dict(), 'weights/Final_LPRNet_model.pth')
    
    STN = STNet()
    STN.to(device)
    # STN.load_state_dict(torch.load('weights/STN_model_Init.pth', map_location=lambda storage, loc: storage))
    # STN.load_state_dict(torch.load('weights/Final_STN_model.pth', map_location=lambda storage, loc: storage))
    checkpoint = torch.load('saving_ckpt/stn_Iter_000900_model.ckpt')
    STN.load_state_dict(checkpoint['net_state_dict'])
    STN.eval()
    print("STN loaded")
    
#    torch.save(STN.state_dict(), 'weights/Final_STN_model.pth')
    
    dataset = LPRDataset([args.img_dirs], args.img_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn) 
    print('dataset loaded with length : {}'.format(len(dataset)))
    
    ACC = eval(CLASS, lprnet, STN, dataloader, dataset, device)
    print('the accuracy is {:.2f} %'.format(ACC*100))
    
    visualize_stn()

