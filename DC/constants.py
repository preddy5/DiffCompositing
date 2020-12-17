import math
import os

from DC.common import make_tensor
from DC.config import args
from DC.scales import all_scales

# matplotlib.use('Agg')
# ------------------------- variables -------------------------
cwd = os.getcwd()
pattern = args.pattern
n_elements = 1
n_soft_elements = int(len(all_scales[pattern])/2)
folder_root = args.folder
if folder_root == '':
    folder_root = cwd#os.path.dirname(cwd)

element_filename = []
for i in range(n_soft_elements):
    element_filename.append('{}/data/{}/elements/{}.png'.format(folder_root, pattern, i+1))
pattern_filename = '{}/data/{}/pattern.png'.format(folder_root, pattern)
folder = "{}/logs/expansion/{}_{}/".format(folder_root, pattern, args.version)

non_white_background = args.non_white

# number = [4, 4]
scale_x_single = all_scales[pattern][0::2]
scale_y_single = all_scales[pattern][1::2]
# ------------------------- calculated constants -------------------------
base_resize = [args.base_size, args.base_size]
expand_size = [1, 3, base_resize[0], base_resize[1]]
if args.expand:
    expand_size = [1, 3, base_resize[0]*2, base_resize[0]*2]

number = [int(1/(scale_y_single[0]* base_resize[1]/expand_size[3])), int(1/(scale_x_single[0]* base_resize[0]/expand_size[2]))]
number[0] = args.sample#(int(number[0] / n_elements)+1) * n_elements
number[1] = args.sample#(int(number[1] / n_elements)+1) * n_elements

repeats = int(number[0] * number[1] / n_elements)
print('Number of elements:', number)
expand_size[0] = repeats

# ------------------------- initializations and constants -------------------------
lr = args.lr
iter = args.num_iter
LBFGS = False
multi = True
soft = args.soft
smooth = False
init_iter = 100000
resume = args.resume
d_threshold = (math.sqrt((max(scale_x_single) * base_resize[0]) ** 2 + (max(scale_y_single) * base_resize[1]) ** 2) -20) / 2


mean = make_tensor([0.485, 0.456, 0.406, 0])[None,:,None, None]
std = make_tensor([0.229, 0.224, 0.225, 1])[None,:,None, None]

content_layers = ['pool_4', 'relu4_2',]# 'relu5_2']
content_weights = {'pool_4': 1.0, 'relu4_2': 1.0,}# 'relu5_2': 1.0}
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4']#, 'conv_5']
style_weights = {'conv_1': 0.2, 'conv_2': 0.2, 'conv_3': 0.2, 'conv_4': 0.2}#, 'conv_5': 200}
hist_layers = ['relu4_2', 'conv_1']
hist_weights = {'relu4_2': 1.0, 'conv_1': 1.0}
