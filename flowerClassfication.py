import imageio
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# from flowerData import flower
from torch import optim
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
plt.rcParams['font.sans-serif']=['STSong']
import numpy as np
import torch.nn.functional as F
from resnet import ResNet
import json
import scipy.misc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, loader):
    correct=0
    total = len(loader.dataset)
    model.eval()
    for x, y in loader:
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()

    return correct / total

# def train():
#
#     model = ResNet18(5).to(device)
#     optimizer = optim.SGD(model.parameters(), lr=float(args.learning_rate), momentum=0.9)
#     criteon = nn.CrossEntropyLoss()
#
#     global_step = 0
#     best_acc, best_epoch = 0, 0
#     acc_list = []
#     for epoch in range(epochs):
#
#         for step, (x, y) in enumerate(train_loader):
#             x,y = x.to(device), y.to(device)
#             model.train()
#             logits = model(x)
#             loss = criteon(logits, y)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             print(step, "loss:", loss.item())
#
#             global_step += 1
#
#         val_acc = evaluate(model, val_loader)
#         acc_list.append(val_acc)
#         if val_acc > best_acc:
#             best_epoch = epoch
#             best_acc = val_acc
#             torch.save(model.state_dict(), 'savedata/flower.mdl')
#
#         print('epochs', epochs,'best acc:', best_acc, 'best epoch', best_epoch)
#
#     model.load_state_dict(torch.load('savedata/flower.mdl'))
#     model.eval()
#     test_acc = evaluate(model, test_loader)
#     print('test acc', test_acc)
#     plot_curve(acc_list)


def get_image(image_dir):
    trans = transforms.Compose([
        lambda x: Image.open(x).convert('RGB'),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = trans('img/ju.jpg')
    image = torch.unsqueeze(image, dim=0)
    return image


def get_k_layer_feature_map(model_layer, k, x):
    with torch.no_grad():
        for index, layer in enumerate(model_layer):  #model的第一个Sequential()是有多层，所以遍历
            x = layer(x)                             #torch.Size([1, 64, 55, 55])生成了64个通道
            if k == index:
                return x


def show_feature_map(feature_map):
    # feature_map=torch.Size([1, 64, 55, 55]),feature_map[0].shape=torch.Size([64, 55, 55])
            # feature_map[2].shape     out of bounds
    feature_map = feature_map.squeeze(0)  # 压缩成torch.Size([64, 55, 55])

    # 以下4行，通过双线性插值的方式改变保存图像的大小
    feature_map = feature_map.view(1, feature_map.shape[0], feature_map.shape[1], feature_map.shape[2])  # (1,64,55,55)
    upsample = torch.nn.UpsamplingBilinear2d(size=(256, 256))  # 这里进行调整大小
    feature_map = upsample(feature_map)
    feature_map = feature_map.view(feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])

    feature_map_num = feature_map.shape[0]  # 返回通道数
    row_num = np.ceil(np.sqrt(feature_map_num))  # 8
    plt.figure()
    for index in range(1, feature_map_num + 1):  # 通过遍历的s方式，将64个通道的tensor拿出

        plt.subplot(row_num, row_num, index)
        # plt.imshow(feature_map[index - 1], cmap='gray')  # feature_map[0].shape=torch.Size([55, 55])
        plt.imshow(transforms.ToPILImage()(feature_map[index - 1]))
        # 将上行代码替换成，可显示彩色 plt.imshow(transforms.ToPILImage()(feature_map[index - 1]))#feature_map[0].shape=torch.Size([55, 55])
        plt.axis('off')
        imageio.imsave('visual//feature_map_save//' + str(index) + ".jpg", feature_map[index - 1])
    plt.show()



# def visual():
#     k=0
#     image = get_image('img/ju.jpg')
#     model = ResNet18(5)
#     model.load_state_dict(torch.load('savedata/flower.mdl', map_location=torch.device('cpu')))
#     model.eval()
#     model_layer = list(model.children())
#     model_layer = model_layer[0]  # 选择第1层
#     feature_map = get_k_layer_feature_map(model_layer, k, image)
#     show_feature_map(feature_map)



def apply(filename,n=3):
    # 应用
    with open('cat_to_name.json.', 'r') as f:
        flower_name = json.load(f)

    model = ResNet(nclass=102)
    model.set_type()
    model.load_state_dict(torch.load('savedata/flower102.pth', map_location=torch.device('cpu')))
    model.eval()
    trans = transforms.Compose([
        lambda x: Image.open(x).convert('RGB'),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = trans(filename)
    # img = cv2.imread("img/ju.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = []
    image = torch.unsqueeze(image,dim=0)
    with torch.no_grad():
        out = model(image)
    print(out.shape)
    out = nn.Softmax(dim=-1)(out.squeeze(dim=0))
    # pred = out.argmax(dim=1)

    # print(pred.item())
    # print(name[pred.item()])
    # _, pred = out.sort(dim=0, descending=True)
    p, pred_idx = out.topk(n, dim=0, largest=True, sorted=True)
    print(pred_idx.shape)
    for index in pred_idx:
        result.append(flower_name[str(index.item()+1)])

    return result, p.tolist()

    # cv2.putText(img,name[pred.item()],(20,100),cv2.FONT_HERSHEY_TRIPLEX,1.0,(255,0,0),1)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()

if __name__ == '__main__':
    # train()
    result, p = apply('img//73.jpg')
    print(result)
    print(p)
    # visual()