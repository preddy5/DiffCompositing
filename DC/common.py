
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.hub import load_state_dict_from_url
from torchvision import models, transforms
from torchvision.models import VGG


# from optimization.roipoly import RoiPoly
# matplotlib.use('Qt5Agg')

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
    X = np.array(fig.canvas.renderer.buffer_rgba())
    return X


def make_tensor(x, grad=False):
    x = torch.tensor(x, dtype=torch.float32)
    x.requires_grad = grad
    return x


def visualize(img, idx=0):
    k = img.to("cpu", torch.double).data.numpy()[idx,:,:,:]
    n, img_h, img_w = k.shape
    fig, ax = plt.subplots()
    if n==3:
        temp = np.zeros([img_h,img_w,3])
        temp[:,:,0]= k[0,:,:]
        temp[:,:,1]= k[1,:,:]
        temp[:,:,2]= k[2,:,:]
        temp[temp>1] = 1.0
        ax.imshow(temp, vmin=0, vmax=1)
        return fig, ax
    else:
        ax.imshow(k[0, :, :], vmin=0, vmax=1)
        return fig, ax

def tensor2img(tensor, idx=0):
    k = tensor.to("cpu", torch.double).data.numpy()[idx, :, :, :]
    n, img_h, img_w = k.shape
    if n >= 3:
        temp = np.zeros([img_h, img_w, n])
        for i in range(n):
            temp[:, :, i] = k[i, :, :]
        temp[temp > 1] = 1.0
        return temp
    else:
        return k[0, :, :]


def itot(img):
    # Rescale the image
    return img
    H, W, C = img.shape
    image_size = tuple([int((float(MAX_IMAGE_SIZE) / max([H, W])) * x) for x in [H, W]])

    itot_t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        # transforms.ToTensor()
    ])

    # Subtract the means
    normalize_t = transforms.Normalize([103.939, 116.779, 123.68], [1, 1, 1])
    tensor = normalize_t(itot_t(img) * 255)

    # Add the batch_size dimension
    tensor = tensor.unsqueeze(dim=0)
    return tensor


def ttoi(tensor):
    # Add the means
    ttoi_t = transforms.Compose([
        transforms.Normalize([-103.939, -116.779, -123.68], [1, 1, 1])])

    # Remove the batch_size dimension
    tensor = tensor.squeeze()
    img = ttoi_t(tensor)
    img = img.cpu().numpy()

    # Transpose from [C, H, W] -> [H, W, C]
    img = img.transpose(1, 2, 0)
    return img

def downsample(img, scale):
    img = F.avg_pool2d(img, [scale, scale])
    return img #F.upsample(img, scale_factor=[scale, scale])


def upsample(img, scale):
    return F.upsample(img, scale_factor=[scale, scale], mode='bilinear')


def get_vgg():
    vgg = models.vgg19(pretrained=True)
    # vgg.load_state_dict(torch.load('./vgg19-d01eb7cb.pth'), strict=False)
    model = vgg.features.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=0, padding_mode='zero')
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                # layers += [conv2d, nn.ReLU(inplace=True)]
                layers += [conv2d, nn.LeakyReLU(inplace=True, negative_slope=0.01)]
                # layers += [conv2d, nn.SELU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def custom_vgg():
    pretrained_url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
    # config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
    config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
    model = VGG(make_layers(config, batch_norm=False))
    state_dict = load_state_dict_from_url(pretrained_url,
                                          progress=True)
    model.load_state_dict(state_dict)
    model = model.features.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

mse_loss = torch.nn.MSELoss()

def gram(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features.clone(), features.t().clone())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


def content_loss(g, c):
    loss = mse_loss(g, c)
    return loss


def mse_loss_(g, s):
    # c1, c2 = g.shape
    loss = mse_loss(g, s)
    return loss#/ (c1 ** 2)   # Divide by square of channels


def tv_loss(c):
    x = c[:, :, 1:, :] - c[:, :, :-1, :]
    y = c[:, :, :, 1:] - c[:, :, :, :-1]
    loss = torch.sum(torch.abs(x)) + torch.sum(torch.abs(y))
    return loss


def nearest_downsample(tensor, size):
    return torch.nn.functional.interpolate(tensor, size, mode='nearest')


def rotate(X, Y, theta):
    s = torch.sin(theta)
    c= torch.cos(theta)
    return X * c - Y * s, X * s + Y * c


def get_features(model, tensor, max_layers=26):
    layers = {'0': 'conv_1',
     '5': 'conv_2',
     '10': 'conv_3',
     '19': 'conv_4',
     '22': 'relu4_2',  ## content representation
     '31': 'relu5_2',
     '28': 'conv_5',
     '9': 'pool_4',
     '27': 'pool_12',
     }
    layers =  {
        '3': 'conv_1',  # Style layers
        '8': 'conv_2',
        '17': 'conv_3',
        '26': 'conv_4',
        # '35': 'conv_5',
        '9': 'pool_4',
        '22': 'relu4_2',  # Content layers
        # '31' : 'relu5_2'
    }

    # Get features
    features = {}
    x = tensor
    for name, layer in model._modules.items():
        if int(name)>max_layers:
            continue
        x = layer(x)
        if name in layers:
            # if (name == '0'):  # relu4_2
            #     features[layers[name]] = x
            # elif (name == '5'):  # relu4_2
            #     features[layers[name]] = x
            # elif (name == '10'):  # relu4_2
            #     features[layers[name]] = x
            # elif (name == '19'):  # relu5_2
            #     features[layers[name]] = x
            if (name == '22'):  # relu4_2
                features[layers[name]] = x
            elif (name == '31'):  # relu5_2
                features[layers[name]] = x
            # elif (name == '28'):  # relu5_2
            #     features[layers[name]] = x
            elif (name == '9'):  # relu5_2
                features[layers[name]] = x
            elif (name == '27'):  # relu5_2
                features[layers[name]] = x
            else:
                b, c, h, w = x.shape
                features[layers[name]] = [gram(x), x.mean()] #/ (h * w)

            # Terminate forward pass
            # if (name == '35'):
            #     break
    else:
        return features


def img_background_padding(img, background):
    w, h, c = img.shape
    resized = cv2.resize(img, (int(h/2), int(w/2)), interpolation=cv2.INTER_LINEAR)
    top = int((h - int(h/2))/2)
    bottom = h - int(h/2) - top
    left = int((w - int(w/2))/2)
    right = w - int(w/2) - left
    background = background*255
    new_im = cv2.copyMakeBorder(resized, left, right, top, bottom, cv2.BORDER_CONSTANT,
                                value=background)
    return new_im


def pad_image(img, target_dim):
    return


def standardize_multi_elements(all_elements):
    h =0; w=0;
    for i in all_elements:
        new_h, new_w, _ = i.shape
        if new_h > h:
            h = new_h
        if new_w > w:
            w = new_w
    new_all_elements = []
    for i in all_elements:
        new_all_elements.append(pad_image(i, [h, w]))


def final_composite(layers, layer_masks, background):
    n2, _, _, _ = layer_masks.shape
    mask = torch.prod(layer_masks, dim=0, keepdim=True)
    render_t = layers.sum(dim=0).unsqueeze(0) + mask * background
    return render_t

def residual_image(gt, layer_masks, background, layer_loss, threshold, idx=None):
    # select_mask = torch.ones_like(layer_loss)
    # select_mask[layer_loss < threshold] = 0
    layer_masks_ = layer_masks.clone().detach()
    if idx is not None:
        layer_masks_[~idx]  = 1
    else:
        layer_masks_[layer_loss > threshold] = 1
    mask = torch.prod(layer_masks_, dim=0, keepdim=True)
    residual = gt *mask + (1-mask) * background
    return residual

def selectROI(main_pattern_np, poly=True):
    if poly:
        fig = plt.figure()
        plt.imshow(main_pattern_np)
        roi = RoiPoly(color='r', fig=fig)
        mask = roi.get_mask(main_pattern_np)
        mask = mask*1  # changing bool to 0,1
        main_pattern_np_rgba = np.concatenate([main_pattern_np, mask[:,:,None]], axis=-1)
        points = [min(roi.x), min(roi.y), max(roi.x), max(roi.y)]
        element_crop = main_pattern_np_rgba[int(points[1]):int(points[3]), int(points[0]):int(points[2])]
        ##
        roi = cv2.selectROI(main_pattern_np)
        background_crop = main_pattern_np[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        background = background_crop.mean(axis=(0, 1)) / 255
        ##
        # background = (element_crop[:,:,:3] * (1-element_crop[:,:,3:])).mean(axis=(0, 1))/255
        threshold = 0.5 * np.absolute((element_crop[:,:,:3]*element_crop[:,:,3:]) - background*255)/255
        element_crop[:, :, 3:] = element_crop[:,:,3:] * 255
        return element_crop, background, threshold.mean()
    else:
        roi = cv2.selectROI(main_pattern_np)
        element_crop = main_pattern_np[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        threshold = 0.2 * element_crop[:, :, 3:].mean()
        roi = cv2.selectROI(main_pattern_np)
        background_crop = main_pattern_np[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        background = background_crop.mean(axis=(0, 1)) / 255
        return element_crop, background, threshold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def selectBackground(main_pattern_np):
    roi = cv2.selectROI(main_pattern_np)
    cv2.destroyAllWindows()
    background_crop = main_pattern_np[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    background = background_crop.mean(axis=(0, 1)) / 255
    return background

def make_batch(t):
    return t.unsqueeze(0)

def get_IMG_tensor_transform(crop=False):
    transforms_list = []
    transforms_list += [
        transforms.ToTensor(),
        # transforms.RandomCrop((102,102), pad_if_needed =True, fill=0), transforms.Normalize((0.5,), (0.5,)), torchvision.transforms.RandomVerticalFlip(p=0.5), torchvision.transforms.RandomHorizontalFlip(p=0.5),
    ]
    return transforms.Compose(transforms_list)

def vgg_normalize(tensor, mean, std):
    return (tensor-mean)/std

def vgg_renormalize(tensor, mean, std):
    return tensor*std + mean

def location_scale_matrix(slices, device, grad=False):
    if type(slices)==list:
        iter_x, iter_y = slices
        scale_x = 1 / slices[0]
        scale_y = 1 / slices[1]
    else:
        iter_x, iter_y = slices, slices
        scale_x = 1 / slices
        scale_y = 1 / slices

    center = 0  # scale/2
    matrix = []
    for i in range(iter_x):
        for j in range(iter_y):
            matrix.append([i * scale_x + center, j * scale_y + center])
    return torch.tensor(matrix, requires_grad=grad, device=device)#.type(torch.FloatTensor)


def get_grid(w, h, device):
    # using from torch sample
    grid_size = [w, h]
    return torch.from_numpy(np.indices(grid_size).reshape((len(grid_size), -1)).T).type(torch.FloatTensor).to(device)


def remove_invisible(c_variables):
    visibility = torch.nn.functional.sigmoid(c_variables[5])
    new_c_variables = []
    idx = [visibility>0.8]
    for i in c_variables:
        new_c_variables.append(i[idx])
    return new_c_variables

def remove_invisible_z(c_variables, threshold=600):
    visibility = torch.exp(c_variables[6] / 25)
    visibility = visibility.to("cpu", torch.double).data.numpy()
    new_c_variables = []
    idx = [visibility>threshold]
    for i in c_variables:
        new_c_variables.append(i[idx])
    return new_c_variables

def remove_invisible_occluded(c_variables, occlusion_mean, threshold=0.5):
    new_c_variables = []
    idx = [occlusion_mean[:,0]>threshold]
    for i in c_variables:
        new_c_variables.append(i[idx])
    return new_c_variables


def remove_invisible_color(c_variables, threshold=0.1):
    visibility = torch.exp(c_variables[6] / 25)
    visibility = visibility.to("cpu", torch.double).data.numpy()
    new_c_variables = []
    idx = [visibility>threshold]
    for i in c_variables:
        new_c_variables.append(i[idx])
    return new_c_variables

def remove_invisible_elements(c_variables):
    visibility = torch.nn.functional.sigmoid(c_variables[7]).to("cpu", torch.double).data.numpy()
    new_c_variables = []
    idx = np.all([visibility>0.8], axis=0)
    for i in c_variables:
        new_c_variables.append(i[idx[:,0]])
    return new_c_variables


def remove_invisible_box(c_variables, parameter_idx, threshold, less=True, xy_scale=10):
    position = c_variables[parameter_idx]/xy_scale
    new_c_variables = []
    if less:
        idx = [position > threshold]
    else:
        idx = [position<threshold]
    for i in c_variables:
        new_c_variables.append(i[idx])
    return new_c_variables

def hardmax(variable):
    var_softmax = F.softmax(variable, dim=-1)
    variable[var_softmax>0.5] = 100
    variable[var_softmax <= 0.5] = -100
    return variable

def d2_distance_matrix(dpos):
    # dt = torch.transpose(dpos, 1, 2)
    shape = dpos.shape
    dt_rep = dpos[:, None, :].repeat(1, shape[0], 1)
    dt_rep_t = dt_rep.transpose(0, 1)
    sub_dt = dt_rep_t - dt_rep
    sq_sub_dt = sub_dt ** 2
    dist_matrix = sq_sub_dt[:, :, 0] + sq_sub_dt[:, :, 1]
    return dist_matrix + torch.eye(shape[0]).float().to(dist_matrix.device)*100

def chunk_it_up(c_variables):
    new_c_variables = []
    n = len(c_variables[0])
    for idx in range(len(c_variables)):
        new_c_variables.append(list(c_variables[idx].chunk(n, dim=0)))
    return new_c_variables

def select_distance_matrix_idx(distance_matrix, element_classes, visibility, threshold=0.052, min_dist= 2/256):
    idx =[0, ]
    shape = distance_matrix.shape
    distance_matrix = torch.sqrt( distance_matrix).to("cpu", torch.double).data.numpy()
    element_classes = element_classes.to("cpu", torch.double).data.numpy()
    visibility = visibility.to("cpu", torch.double).data.numpy()
    for i in range(1, shape[0]):
        if type(threshold)==list:
            threshold_class = threshold[int(element_classes[i])]
        if np.all(distance_matrix[:, i][idx]>threshold_class):
            idx.append(i)
        elif np.all(element_classes[i]!=element_classes[idx][distance_matrix[:, i][idx]<=threshold_class]) and np.all(distance_matrix[:, i][idx]>min_dist):
            idx.append(i)
    return idx

def remove_bgcolor_elements(c_variables, color_values, bg_color, threshold):
    bg_color = bg_color.to("cpu", torch.double).data.numpy()
    distance = np.sqrt(((color_values - bg_color[:,:,0,0]) ** 2).sum(axis=1))
    for i in range(distance.shape[0]):
        print(i, distance[i])
    new_c_variables = []
    idx = [distance>threshold]
    for i in c_variables:
        new_c_variables.append(i[idx])
    return new_c_variables


def sort_by_z(c_variables, inverse=True):
    new_c_variables = []
    z = c_variables[6].to("cpu", torch.double).data.numpy()
    if inverse:
        idx = np.argsort(-z)
    else:
        idx = np.argsort(z)
    for i in c_variables:
        new_c_variables.append(i[idx])
    return new_c_variables


def location_scale_matrix(slices, device, grad=False):
    if type(slices)==list:
        iter_x, iter_y = slices
        scale_x = 1 / slices[0]
        scale_y = 1 / slices[1]
    else:
        iter_x, iter_y = slices, slices
        scale_x = 1 / slices
        scale_y = 1 / slices

    center = 0  # scale/2
    matrix = []
    for i in range(iter_x):
        for j in range(iter_y):
            matrix.append([i * scale_x + center, j * scale_y + center])
    return torch.tensor(matrix, requires_grad=grad, device=device)#.type(torch.FloatTensor)