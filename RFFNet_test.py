import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from model.RFFNet4 import RFFNet
from data import test_dataset
from datetime import datetime
import time
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path', type=str, default='../dataset/RGBD_for_test/',help='test dataset path')
parser.add_argument('--test_model', type=str, default='./cpts/RFFNet4_cpts/epoch_best.pth',help='test model path')
parser.add_argument('--save_path', type=str, default='./cpts/RFFNet4_cpts/',help='test model path')

opt = parser.parse_args()

dataset_path = opt.test_path
model_path = opt.test_model
save_dir = opt.save_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
elif  opt.gpu_id=='2':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    print('USE GPU 2')
elif  opt.gpu_id=='3':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    print('USE GPU 3')

#load the model
model = RFFNet()
model.load_state_dict(torch.load(model_path))

print(sum([x.nelement() for x in model.parameters()])/1024/1024)

def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


def visualize_feature_map(img_batch,img_name,num):
    #print(img_batch.size()[0:])

    feature_map = torch.squeeze(img_batch, 0)
    #print(feature_map.shape)
    if(len(feature_map.size())==2):
        feature_map = torch.unsqueeze(feature_map,0)


    feature_map_combination = []
    num_pic = feature_map.shape[0]
    row, col = get_row_col(num_pic)

    for i in range(0, num_pic):
        feature_map_split = feature_map[i, :, :]
        feature_map_combination.append(feature_map_split)

    feature_map_sum = sum(ele for ele in feature_map_combination)
    feature_map_sum = feature_map_sum.cuda().data.cpu()
    plt.imshow(feature_map_sum)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig('/root/workspace/save/'+img_name+'_'+str(num)+".png", bbox_inches='tight', dpi=18, pad_inches=0.0)

model.cuda()
model.eval()
total_time = 0
total_image = 0
#test
test_datasets = ['NJU2K', 'NLPR', 'STERE', 'DES', 'SIP']#,'DUT']
for dataset in test_datasets:
    save_path = save_dir + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    depth_root = dataset_path + dataset + '/depth/'
    gt_root = dataset_path + dataset + '/GT/'
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    total_image += test_loader.size
    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            res, _, _, _, _ = model(image, depth)
            torch.cuda.synchronize()
            total_time += time.time() - start_time
            #print(time.time() - start_time)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ', save_path+name)
        cv2.imwrite(save_path+name, res*255)
    print('Test Done!')
print('total image:', total_image)
fps = total_image / total_time
print('FPS: ', fps)
