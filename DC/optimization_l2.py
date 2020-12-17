import json
import time

import torch
from distutils.dir_util import copy_tree

from DC.common import vgg_normalize, vgg_renormalize, d2_distance_matrix, select_distance_matrix_idx, \
    sort_by_z, custom_vgg, remove_invisible_z
from DC.compositing import composite_layers
from DC.constants import *
from DC.loss import loss_fn_l2
from DC.utils import load_element, load_base_pattern, init_element_pos, render_constants, make_cuda, init_optimizer, \
    save_tensor, get_ss_l2, select_op_variables, save_tensor_z

# -------------------------  -------------------------
if not resume:
    copy_tree(cwd+ '/DC', folder+ '/DC')

    with open(folder+'hyperparameters.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        print('FOLDER ALREADY EXISTS')

# -------------------------  -------------------------
w = 1
w1 = 1
w2 = 0.5
w3 = 0.5
w_o = 0
tau =1.5
remove_things = False
num_seeds = 1

def main():
    global soft, w, w1, w2, w3, w_o, init_iter, base_resize, expand_size, n_soft_elements, tau, remove_things, num_seeds
    if args.complex_background:
        base_pattern, background, background_img = load_base_pattern(pattern_filename, args.tiled, base_resize, blur=False, complex_background=True)
        background_img = background_img.cuda()
    else:
        base_pattern, background = load_base_pattern(pattern_filename, args.tiled, base_resize, blur=False)
        background_img = None
    elements = load_element(element_filename, size=base_resize)
    parameters = init_element_pos(number)
    constants = render_constants(number[0]*number[1], scale_x_single, scale_y_single, base_resize, n_soft_elements, expand_size, color=args.color)
    base_pattern, background, mean_c, std_c = make_cuda([base_pattern, background, mean, std])
    base_pattern = vgg_normalize(base_pattern, mean_c[:,:3], std_c[:,:3])
    c_variables = make_cuda(parameters+constants)
    init_c_variables = []
    for i in c_variables:
        init_c_variables.append(i.clone().detach())
    op_variables = select_op_variables(n_elements, args, c_variables)

    if args.tiled:
        base_resize = [512, 512]
    background.requires_grad = True
    if non_white_background:
        op_variables = op_variables+[background, ]
    optimizer, scheduler = init_optimizer(op_variables, args.lr)

    # ----------------- remove unnecessary ------------------------------
    def remove_unecessary(c_variables):
        if not args.color:
            c_variables = remove_invisible_z(c_variables, 2000)
        c_variables = sort_by_z(c_variables)
        element_classes = torch.argmax(c_variables[7], dim=1)
        distance_matrix = d2_distance_matrix(torch.cat([c_variables[0][:, None]/10, c_variables[1][:, None]/10], dim=1))
        idx = select_distance_matrix_idx(distance_matrix, element_classes, c_variables[5], [10/256, 20/256, 10/256, 10/256, 20/256, 15/256, 15/256, 30/256, 30/256], min_dist=10/256)
        new_c_variables = []
        for i in c_variables:
            new_c_variables.append(i[idx])
        c_variables = make_cuda(new_c_variables)
        op_variables = select_op_variables(n_elements, args, c_variables)
        if non_white_background:
            op_variables = op_variables + [background, ]
        optimizer, scheduler = init_optimizer(op_variables, args.lr*0.1)
        return c_variables, optimizer, scheduler
    # ----------------- init ------------------------------
    c_variables[6].requires_grad = False
    if resume:
        load_iter = args.resume_int
        PATH = folder + 'checkpoint_' + str(load_iter)
        checkpoint = torch.load(PATH)
        c_variables = checkpoint['c_variables']
        background = checkpoint['background']
        if args.seed:
            c_variables_1 = c_variables.copy()
            c_variables_1[6] = c_variables_1[6] / 2
            c_variables_2 = []
            for c in init_c_variables:
                c_variables_2.append(c.clone().detach())
            c_variables = []
            for idx in range(len(c_variables_1)):
                c_variables.append(torch.cat([c_variables_1[idx], c_variables_2[idx], ]))
        c_variables = make_cuda(c_variables)
        torch.cuda.empty_cache()
        op_variables = select_op_variables(n_elements, args, c_variables)
        background.requires_grad = True
        if non_white_background:
            op_variables = op_variables + [background, ]
        optimizer, scheduler = init_optimizer(op_variables, args.lr)
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler = CyclicLR(optimizer, base_lr=lr, max_lr=lr * 4, cycle_momentum=False, mode='exp_range',
        #                      step_size_up=1500)
        # ----------------------declare variables -------------------
        init_iter = load_iter-1
        w = 1
        w1 = 1
        w2 = 1
        w3 = 1.0
        w_o = 0
        # tau = tau + 10
        if args.layers:
            c_variables[6].requires_grad = True

    current_soft = False
    base_pattern = vgg_renormalize(base_pattern, mean_c[:, :3], std_c[:, :3])
    save_tensor(folder, 'BASE', base_pattern)
    ss = get_ss_l2(base_pattern)

    start_time = time.time()
    for i in range(init_iter+1, init_iter + iter):
        # tau = 0.1
        render, overlap = composite_layers(elements, c_variables, background, n_elements, expand_size, current_soft, tau, color=args.color, background_img=background_img)
        ws = [w, w1, w2, w3, w_o]
        loss = loss_fn_l2(render, overlap, ss, ws) + overlap*w_o
        if i%10==0:
            save_tensor(folder, str(i), render)
            print('iteration {} '.format(i), (loss).to('cpu'), (overlap*(w_o)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## Everything below this is to improve convergence
        if (i- 100000)==500 :
            w_o = 0#(10**-7)
            if args.layers:
                c_variables[6].requires_grad = True
        if i%3000==0:
            # tau = tau +1
            print(torch.exp(c_variables[6]/25))
            if args.layers:
                c_variables[6].requires_grad = True
            save_tensor_z(folder, str(i)+'_z', render, c_variables[0]/10, c_variables[1]/10, c_variables[6], size=128)
            save_tensor_z(folder, str(i)+'_idx', render, c_variables[0]/10, c_variables[1]/10, torch.arange(c_variables[6].shape[0]), size=128)
            torch.save({
                'c_variables': c_variables,
                'optimizer_state_dict': optimizer.state_dict(),
                'base_size':base_resize,
                'xy_scale_factor':10,
                'soft':current_soft,
                'background':background}, folder+'checkpoint_'+str(i))
            w1 = min(w1 + 0.05, 1.5)
            w1 = min(w1 + 0.05, 1.5)
            w2 = max(w2 - 0.05, 0.2)
            w3 = max(w3 - 0.05, 0.2)
        if ((i- 100000)%8000 - 4000 == 500 or  (i- 100000)%8000== 500)and remove_things:
            w_o = 0
            for g in optimizer.param_groups:
                g['lr'] = args.lr
        if (i- 100000)%8000==4000 and remove_things:
            w_o = 0
            elapsed_time = time.time() - start_time
            print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
            print(elapsed_time)
            c_variables, optimizer, scheduler = remove_unecessary(c_variables)
            print(torch.nn.functional.sigmoid(c_variables[5]/10))
            render, overlap = composite_layers(elements, c_variables, background, n_elements, expand_size, current_soft,
                                               tau, color=args.color, background_img=background_img)
            save_tensor(folder, 'postremoval_'+str(i), render)

        if args.seed and num_seeds>0:
            if (i- 100000)%8000==0:
                num_seeds = num_seeds - 1
                torch.save({
                    'c_variables': c_variables,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'base_size': base_resize,
                    'xy_scale_factor': 10,
                    'soft': current_soft,
                    'background': background}, folder + 'checkpoint_' + str(i))
                if args.rotation:
                    c_variables_1 = c_variables.copy()
                    c_variables_1[6] = c_variables_1[6]*2/3
                    # c_variables_1[7] = c_variables_1[7]
                    c_variables_2 = c_variables.copy()
                    c_variables_2[2] = c_variables_2[2] + (90/(1.5*3.14159*57.2958))
                    c_variables_2[6] = c_variables_2[6]*2/3
                    # c_variables_2[7] = torch.ones_like(c_variables_2[7])#/4
                    c_variables_3 = c_variables.copy()
                    c_variables_3[2] = c_variables_3[2] + (180/(1.5*3.14159*57.2958))
                    c_variables_3[6] = c_variables_3[6]*2/3
                    # c_variables_3[7] = torch.ones_like(c_variables_3[7])#/4
                    c_variables_4 = c_variables.copy()
                    c_variables_4[2] = c_variables_4[2] + (270/(1.5*3.14159*57.2958))
                    c_variables_4[6] = c_variables_4[6]*2/3

                    c_variables = []
                    for idx in range(len(c_variables_1)):
                        c_variables.append(torch.cat([c_variables_1[idx], c_variables_2[idx], c_variables_3[idx], c_variables_4[idx]]))
                    c_variables = make_cuda(c_variables)
                    op_variables = select_op_variables(n_elements, args, c_variables)
                    render, overlap = composite_layers(elements, c_variables, background, n_elements, expand_size, current_soft,
                                                       tau, color=args.color, background_img=background_img)
                    save_tensor(folder, 'postseeding_'+str(i), render)

                    optimizer, scheduler = init_optimizer(op_variables+[background, ], args.lr*0.1)
                    remove_things = True
                else:
                    c_variables_1 = c_variables.copy()
                    if args.color:
                        c_variables_1[7] = torch.ones_like(c_variables_1[7])
                        c_variables_2 = []
                        for c in init_c_variables:
                            c_variables_2.append(c.clone().detach())
                        c_variables = []
                        for idx in range(len(c_variables_1)):
                            c_variables.append(torch.cat([c_variables_1[idx], c_variables_2[idx],]))
                    else:
                        c_variables_1[6] = c_variables_1[6] / 2
                        c_variables_2 = []
                        for c in init_c_variables:
                            c_variables_2.append(c.clone().detach())
                        c_variables = []
                        for idx in range(len(c_variables_1)):
                            c_variables.append(torch.cat([c_variables_1[idx], c_variables_2[idx],]))
                    c_variables = make_cuda(c_variables)
                    op_variables = select_op_variables(n_elements, args, c_variables)
                    render, overlap = composite_layers(elements, c_variables, background, n_elements, expand_size, current_soft,
                                                       tau, color=args.color)
                    save_tensor(folder, 'postseeding_'+str(i), render)

                    if non_white_background:
                        op_variables = op_variables + [background, ]
                    optimizer, scheduler = init_optimizer(op_variables, args.lr*0.1)
                    remove_things = True
                w_o = 0



if __name__ == '__main__':
    main()
