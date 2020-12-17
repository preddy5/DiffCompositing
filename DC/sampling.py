import kornia
import torch
import torch.nn.functional as F

from DC.common import get_grid

def downsample(img, scale):
    return F.avg_pool2d(img, scale, count_include_pad=False, stride=[1,1])

def sampling_layer(img, x, y, theta, scale_x, scale_y, n_soft, size, extend_canvas=False, blur=False, blur_kernel=3, background=None, color=False):
    """
    Renders a set to 2D canvas
    :param img: NxCxIHxIW input patch
    :param x: N center location x coordinate system on canvas is 0-1
    :param y: N center location y
    :param scale_x: N scale of the element wrt to canvas x ratio of length w.r.t to total canvas length
    :param scale_y: N scale of the element wrt to canvas y

    """
    batch, c, h, w = img.shape
    repeat = x.shape[0]
    if c==3:
        # adding alpha channel
        img = torch.cat([img, torch.ones([batch, 1, h, w]).to(img.device)], axis=1)
    # ------------------ extend canvas ---
    if extend_canvas:
        x = (x+0.5)/2
        y = (y+0.5)/2
        scale_x = scale_x/2
        scale_y = scale_y/2
    # ------------------ ------------- ---
    scale_x = scale_x / 2
    scale_y = scale_y / 2
    theta = theta * 3.14159
    aff_matrix = torch.stack([torch.cos(theta), -torch.sin(theta), x,
                              torch.sin(theta), torch.cos(theta), y], dim=1).view(-1, 2, 3)
    A_batch = aff_matrix[:, :, :2]
    b_batch = aff_matrix[:, :, 2].unsqueeze(1)

    _coords = get_grid(size[2], size[3], aff_matrix.device)
    coords = _coords.unsqueeze(0).repeat(repeat, 1, 1).float()
    coords[:, :, 0] = coords[:, :, 0] / (size[2] - 1)
    coords[:, :, 1] = coords[:, :, 1] / (size[3] - 1)

    coords = coords - b_batch
    coords = coords.bmm(A_batch.transpose(1, 2))
    sc = torch.stack([scale_x, scale_y], dim=-1)
    coords = coords/ sc[:, None, :]
    # coords[:, :, 0] = scale_coord[:, :, 0] / scale_x[:, None]
    # coords[:, :, 1] = scale_coord[:, :, 1] / scale_y[:, None]

    grid = coords.view(-1, size[2], size[3], 2)
    grid = torch.stack([grid[:, :, :, 1], grid[:, :, :, 0]], dim=-1)
    # ------------------ downsampling important for gradients ---
    n_soft_elements = n_soft.shape[1]
    if not color:
        img_bg = img.clone()
        if background is not None:
            for e in range(0, n_soft_elements):
                img_bg[:,e*4:(e*4+3)] = img[:, e*4:(e*4+3)]*img[:,(e*4+3):(e*4+4)] + (1-img[:,(e*4+3):(e*4+4)])*background
    else:
        img_bg = img
    if blur:
        img = kornia.filters.gaussian_blur2d(img, (blur_kernel, blur_kernel), (blur_kernel, blur_kernel))
    if batch==1:
        img = img_bg.repeat(repeat, 1, 1, 1)
    render =  torch.nn.functional.grid_sample(img, grid, 'bilinear', 'zeros')
    return render
