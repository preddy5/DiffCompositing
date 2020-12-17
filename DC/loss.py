import kornia

from DC.common import get_features, mse_loss_
from DC.constants import style_weights, style_layers, content_layers, content_weights

dsample = kornia.transform.PyrDown()


def loss_fn_l2(render, overlap, ss, ws):
    s_feat, s_feat_1, s_feat_2, s_feat_3 = ss
    w, w1, w2, w3, w_o = ws
    half_x, half_y = int(render.shape[2] * 1 / 2), int(render.shape[3] * 1 / 2)
    g_feat_1 = dsample(render)
    g_feat_2 = dsample(g_feat_1)
    g_feat_3 = dsample(g_feat_2)
    s_loss_1 = mse_loss_(g_feat_1, s_feat_1)#torch.abs(g_feat_1 - s_feat_1).mean()
    s_loss_2 = mse_loss_(g_feat_2, s_feat_2)#torch.abs(g_feat_2 - s_feat_2).mean()
    s_loss_3 = mse_loss_(g_feat_3, s_feat_3)#torch.abs(g_feat_3 - s_feat_3).mean()
    g_feat = render
    s_loss = mse_loss_(g_feat, s_feat)#torch.abs(g_feat - s_feat).mean()
    loss = (s_loss * w + s_loss_1 * w1 + s_loss_2 * w2 + s_loss_3 * w3) + overlap * w_o
    return loss  # + h_loss

