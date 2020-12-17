import PIL
import kornia
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.optim.lr_scheduler import CyclicLR

from DC.common import make_tensor, visualize, get_features, make_batch, get_IMG_tensor_transform, tensor2img, location_scale_matrix, fig2data
from DC.config import args
from DC.constants import expand_size, non_white_background
from DC.scales import all_background

matplotlib.use('Agg')

transforms = get_IMG_tensor_transform()
transforms_element = get_IMG_tensor_transform(True)
dsample = kornia.transform.PyrDown()
def load_element(filename, gaussian=False, rotate=False, size=[256,256], pad=True):
    n_elements = len(filename)
    all_elements = []
    all_t = []
    if gaussian:
        for i in range(n_elements):
            img = np.zeros([4,100,100])
            img[i, 50, 50] = 1
            img[3, 50, 50] = 1
            elements = make_tensor(img)
            elements = make_batch(elements)
            all_t.append(elements)
        all_elements.append(torch.cat(all_t, dim=1).cuda())
        return all_elements
    if rotate:
        angles =[0, 90, 180, 270]
    else:
        angles = [0,]
    for i in filename:
        im = Image.open(i).convert('RGBA')
        w, h = im.size
        im = im.resize((int(w * size[0] / 256), int(h * size[1] / 256)), resample=PIL.Image.LANCZOS)
        w, h = im.size
        desired_size = int(100*size[0]/256)
        print(w, h, desired_size)
        if pad:
            if w>desired_size or h> desired_size:
                im = im.resize((desired_size, desired_size), resample=PIL.Image.LANCZOS)
            if w<desired_size or h<desired_size:
                delta_w = desired_size - w
                delta_h = desired_size - h
                padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
                im = ImageOps.expand(im, padding)
        for a in angles:
            im_rotated = im.rotate(a)
            elements = transforms_element(im_rotated)
            elements = make_batch(elements)

            all_t.append(elements)
    if not pad:
        return all_t
    all_elements.append(torch.cat(all_t, dim=1).cuda())
    return all_elements

def load_element_PIL(filename, size=[256,256], padding=True):
    n_elements = len(filename)
    print(n_elements)
    all_t = []
    for i in filename:
        im = Image.open(i).convert('RGBA')
        w, h = im.size
        im = im.resize((int(w * size[0] / 256), int(h * size[1] / 256)), resample=PIL.Image.LANCZOS)
        w, h = im.size
        desired_size = 100#int(100 * size[0] / 256)
        print('Element shape:', w, h, desired_size)
        if padding:
            if w>desired_size or h> desired_size:
                im = im.resize((desired_size, desired_size))
            if w<desired_size or h<desired_size:
                delta_w = desired_size - w
                delta_h = desired_size - h
                padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
                im = ImageOps.expand(im, padding)
        all_t.append(im)
    return all_t

def tile_exemplar(tile):
    expanded_pattern = torch.ones([1, expand_size[1], expand_size[2], expand_size[3]]).cuda()
    expanded_pattern[:, :, 256:, 256:] = tile
    expanded_pattern[:, :, :256, 256:] = tile
    expanded_pattern[:, :, 256:, :256] = tile
    expanded_pattern[:, :, :256, :256] = tile
    return expanded_pattern

def load_base_pattern(filename, tiled=False, size=(256,256), blur =False, complex_background=False):
    image = Image.open(filename).convert('RGB').resize(size, resample=PIL.Image.LANCZOS)
    background = np.array([1.,1.,1.])
    if args.pattern in all_background.keys():
        background = np.array(all_background[args.pattern])
        print(background)
    background = make_tensor(background, False)[None, :, None, None]
    if non_white_background:
        background = np.array(image).mean(axis=0).mean(axis=0)/255#selectBackground( np.array(image))
        background = make_tensor(background, False)[None, :, None, None]
    base_pattern = transforms(image)
    base_pattern = make_batch(base_pattern)
    if complex_background:
        filename_background = filename.replace('pattern.png', 'background.png')
        background_img = Image.open(filename_background).convert('RGB').resize(size, resample=PIL.Image.LANCZOS)
        background = np.array(background_img).mean(axis=0).mean(axis=0) / 255
        background = make_tensor(background, False)[None, :, None, None]
        background_img = transforms(background_img)
        background_img = make_batch(background_img)
        return base_pattern, background, background_img
    if blur:
        base_pattern = kornia.filters.gaussian_blur2d(base_pattern, (3, 3), (3, 3))
    if tiled:
        base_pattern = tile_exemplar(base_pattern)
    return base_pattern, background

def render_constants(shape, scale_x_single, scale_y_single, base_resize, n_soft_elements, expand_size, color=False):
    scale_x = torch.ones(shape)  # np.random.random_sample(x.shape)/7.5#
    scale_y = torch.ones(shape)  # np.random.random_sample(x.shape)/7.5#
    theta = torch.zeros(shape)#*torch.rand(shape)#
    visibility = torch.ones(shape)*2#*20
    n_elements = len(scale_x_single)
    soft_elements = torch.zeros([shape, n_soft_elements])
    layers_z = torch.ones([shape, ])*9*25
    scale_x = scale_x* scale_x_single[0] * base_resize[0]/expand_size[2] #*base_resize[0]/256
    scale_y = scale_y* scale_y_single[0] * base_resize[1]/expand_size[3] #*base_resize[1]/256
    if color:
        rgb = torch.ones([shape, 3])*0.5*20
        return [theta, scale_x, scale_y, visibility, layers_z, soft_elements, rgb]
    return [theta, scale_x, scale_y, visibility, layers_z, soft_elements]


def init_optimizer(parameters, lr):
    # tuple_params = [list(i) for i in parameters]
    tuple_params = list(j for i in parameters for j in (i if isinstance(i, list) else (i,)))
    for i in tuple_params:
        i.requires_grad = True
    if args.LBFGS:
        optimizer = torch.optim.LBFGS(parameters, lr=0.01)
        return optimizer, None
    optimizer = torch.optim.Adam(tuple_params, lr=lr, eps=1e-06, betas=(0.9, 0.90))#, betas=(0.65, 0.70))
    scheduler = CyclicLR(optimizer, base_lr=lr, max_lr=lr*2, cycle_momentum=False, mode= 'exp_range', step_size_up=1500)
    return optimizer, scheduler

def init_element_pos(number):
    lo_scale_matrix = location_scale_matrix(number, 'cpu', True)
    y_t = (lo_scale_matrix[:,1]+0.5/(number[1]+0.5))*10
    x_t = (lo_scale_matrix[:,0]+0.5/(number[0]+0.5))*10
    return [x_t, y_t]

def make_cuda(variables):
    cuda_variables = []
    for i in variables:
        cuda_variables.append(i.clone().detach().cuda())
    return cuda_variables

def visualize_list(tensors):
    for i in tensors:
        visualize(i)
        plt.show()

def save_tensor(folder, name, tensor, idx=0):
    img = tensor2img(tensor, idx)
    img[img>1] = 1
    img[img<0] = 0
    plt.imsave(folder + name + ".png", img)


def save_tensor_z(folder, name, tensor, x, y, z, size=384):
    img = tensor2img(tensor)
    x = (x*size).to("cpu", torch.int).data.numpy()
    y = (y*size).to("cpu", torch.int).data.numpy()
    z = z.to("cpu", torch.double).data.numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    for i in range(x.shape[0]):
        ax.text(x[i], y[i], str('%.1f'%z[i]), fontsize=15, color='green')
    image = fig2data(fig)
    plt.imsave(folder + name + ".png", image)

def save_img_z(folder, name, img, x, y, z, size=384, n=1):
    x = (x*size).to("cpu", torch.int).data.numpy()
    y = (y*size).to("cpu", torch.int).data.numpy()
    z = z.to("cpu", torch.double).data.numpy()
    fig = plt.figure()
    fig.set_size_inches(18.5, 18.5)
    ax = fig.add_subplot(111)
    ax.imshow(img)
    for i in range(x.shape[0]):
        ax.text(x[i], y[i], str('%0.*f'%(n, z[i])), fontsize=12, color='blue')
    image = fig2data(fig)
    plt.imsave(folder + name + ".png", image)

def save_array_z(folder, name, img, x, y, z, size=384, n=3):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    for i in range(x.shape[0]):
        ax.text(x[i], y[i], str("%0.*f"% (n, z[i])), fontsize=10, color='red')
    image = fig2data(fig)
    plt.imsave(folder + name + ".png", image)


def bilinear_downsample(tensor, size):
    return torch.nn.functional.interpolate(tensor, size, mode='bilinear')

def select_op_variables(n_elements, args, c_variables):
    op_variables = []
    for i in range(n_elements):
        # 6 because 6 variables for each compositing
        op_variables.append(c_variables[0])
        op_variables.append(c_variables[1])
        if args.rotation:
            op_variables.append(c_variables[2])
        if args.soft:
            op_variables.append(c_variables[5])
        if args.layers:
            op_variables.append(c_variables[6])
        if args.soft_elements:
            op_variables.append(c_variables[7])
        if args.color:
            op_variables.append(c_variables[8])
    return op_variables

def redo_variables(c_variables, n_elements, repeats):
    variables = []
    for i in range(n_elements):
        variables.append(torch.ones_like(c_variables[0][i::n_elements]) * c_variables[0][i::n_elements])
        variables.append(torch.ones_like(c_variables[1][i::n_elements]) * c_variables[1][i::n_elements])
        variables.append(torch.ones_like(c_variables[2][i * repeats:(i + 1) * repeats]) * c_variables[2][i * repeats:(i + 1) * repeats])
        variables.append(torch.ones_like(c_variables[3][i * repeats:(i + 1) * repeats]) * c_variables[3][i * repeats:(i + 1) * repeats])
        variables.append(torch.ones_like(c_variables[4][i * repeats:(i + 1) * repeats]) * c_variables[4][i * repeats:(i + 1) * repeats])
        variables.append(torch.ones_like(c_variables[5][i * repeats:(i + 1) * repeats]))
        variables.append(torch.ones_like(c_variables[6][i * repeats:(i + 1) * repeats])*5)
        variables.append(torch.ones_like(c_variables[7][i * repeats:(i + 1) * repeats]))
    return variables

def get_ss(model, base_pattern, content_layers):
    with torch.no_grad():
        s_feat_1 = get_features(model, dsample(base_pattern), 17)
        s_feat_2 = get_features(model, dsample(dsample(base_pattern)), 10)#get_features(model, bilinear_downsample(base_pattern,[int(half_x*3/4), int(half_y*3/4)]))
        s_feat_3 = None#get_features(model, bilinear_downsample(base_pattern,[int(half_x/2), int(half_y/2)]))
        s_feat = get_features(model, base_pattern)
        corr_s = {}
        for j in content_layers:
            shape = s_feat[j].shape
            corr_s[j] = deep_corr_matrix(s_feat[j], shape[2], shape[3])
    return [s_feat, s_feat_1, s_feat_2, s_feat_3, corr_s]


def base_downsampled(base_pattern):
    with torch.no_grad():
        half_x, half_y = int(base_pattern.shape[2]*1 / 2), int(base_pattern.shape[3]*1 / 2)
        s_feat_1 = dsample(base_pattern)
        s_feat_2 = dsample(s_feat_1)
        s_feat_3 = dsample(s_feat_2)
        s_feat = base_pattern
    return [s_feat, s_feat_1, s_feat_2, s_feat_3]


def get_ss_l2(base_pattern):
    with torch.no_grad():
        half_x, half_y = int(base_pattern.shape[2]*1 / 2), int(base_pattern.shape[3]*1 / 2)
        s_feat_1 = dsample(base_pattern)
        s_feat_2 = dsample(s_feat_1)
        s_feat_3 = dsample(s_feat_2)
        s_feat = base_pattern
    return [s_feat, s_feat_1, s_feat_2, s_feat_3]
