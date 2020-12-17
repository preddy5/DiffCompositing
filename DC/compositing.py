import torch
import torch.nn.functional as F

from DC.common import d2_distance_matrix
from DC.sampling import sampling_layer

def composite_layers(elements, c_variables, background, n_elements, expand_size, soft, tau=2.5, return_mask=False,
                     blur_kernel=3, xy_scale_factor=10, color=False, occlusion_value = False, background_img=None):
    x = c_variables[0] / xy_scale_factor
    y = c_variables[1] / xy_scale_factor
    render = sampling_layer(elements[0], y, x,
                            c_variables[2] * 1.5,
                            c_variables[3],
                            c_variables[4],
                            n_soft=c_variables[7],
                            size=expand_size, blur=True, blur_kernel=blur_kernel,
                            background=background, color=color)

    render_all = render  # torch.cat(render_all, dim=0)
    layer_z = c_variables[6] /25  # torch.cat(layer_z_, dim=0)
    layer_z = torch.exp(layer_z)
    n_soft_elements = c_variables[7].shape[1]

    y_hard = F.softmax(c_variables[7]*tau)
    n_softmax = y_hard #- y_soft.detach() + y_soft
    element_mask = render_all[:, 3:4, :, :]
    alpha = element_mask * n_softmax[:, 0:1, None, None]
    render_rgb = render_all[:, :3, :, :] * n_softmax[:, 0:1, None, None]  # *inv_mask_discrete

    if color:
        color_scaled = c_variables[8][:, :, None, None]/20
        color_value= torch.max(torch.min(color_scaled, 1+(color_scaled-1)*0.001), 0.001*color_scaled)
        render_rgb = render_rgb * color_value

    for e in range(1, n_soft_elements):
        element_mask = render_all[:, (e * 4 + 3):(e * 4 + 4), :, :]
        alpha = alpha + element_mask * n_softmax[:, e:e + 1, None, None]
        current_rgb = render_all[:, e * 4:(e * 4 + 3), :, :] * n_softmax[:, e:e + 1, None, None]
        if color:
            current_rgb = current_rgb*color_value
        render_rgb = render_rgb + current_rgb
    overlap = F.relu(torch.sum(alpha, dim=0, keepdim=True ) -1).mean()
    weighted_alpha = alpha * layer_z[:, None, None, None]

    background_z = 2000
    sum_weighted_alpha = torch.sum(weighted_alpha, dim=0, keepdim=True)
    sum_alpha = sum_weighted_alpha + background_z

    alpha_hat = weighted_alpha / sum_alpha
    z_mask = background_z/ sum_alpha
    render_rgb = render_rgb * alpha_hat
    render_rgb = torch.sum(render_rgb, dim=0, keepdim=True)
    if type(background_img)== type(None):
        render = (render_rgb + z_mask * background)
    else:
        render = (render_rgb + z_mask * background_img)
    if return_mask:
        return render, 0, z_mask
    elif occlusion_value:
        oclusion_measure = alpha_hat.to("cpu", torch.double).data.numpy()
        mask = oclusion_measure > 0.001
        oclusion_measure_mean = (oclusion_measure * mask).sum(axis=3).sum(axis=2) / mask.sum(axis=3).sum(axis=2)
        return render, overlap, oclusion_measure_mean
    return render, overlap
