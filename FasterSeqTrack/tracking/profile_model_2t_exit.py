import argparse
import torch
from thop import profile
from thop.utils import clever_format
import time
import importlib
from torch import nn
import numpy as np


class EarlyExitEncoder(nn.Module):
    """Profile the encoder through a selected early-exit point."""

    def __init__(self, model, exit_layer, threshold=0.0):
        super().__init__()
        self.model = model
        self.exit_layer = exit_layer
        self.threshold = threshold

    def forward(self, images_list):
        xz, _, _, _ = self.model.inference_encoder(
            images_list, self.exit_layer, self.threshold
        )
        return xz


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='fasterseqtrack',
                        help='training script name')
    parser.add_argument('--config', type=str, default='fasterseqtrack_b256_1t', help='yaml configure file name')
    parser.add_argument('--exit_layer', type=int, default=6,
                        help='encoder layer used for fixed early-exit profiling')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='confidence threshold used for early-exit profiling')
    args = parser.parse_args()

    return args


def get_complexity_MHA(m:nn.MultiheadAttention, x, y):
    """(L, B, D): sequence length, batch size, dimension"""
    d_mid = m.embed_dim
    query, key, value = x[0], x[1], x[2]
    Lq, batch, d_inp = query.size()
    Lk = key.size(0)
    """compute flops"""
    total_ops = 0
    # projection of Q, K, V
    total_ops += d_inp * d_mid * Lq * batch  # query
    total_ops += d_inp * d_mid * Lk * batch * 2  # key and value
    # compute attention
    total_ops += Lq * Lk * d_mid * 2
    m.total_ops += torch.DoubleTensor([int(total_ops)])


def evaluate(model, images_list, xz, input_start, hanning, bs,
             exit_layer=6, threshold=0.0):
    """Compute FLOPs, Params, and Speed"""
    custom_ops = {nn.MultiheadAttention: get_complexity_MHA}
    # Profile the actual early-exit path without changing the model's forward method.
    encoder = EarlyExitEncoder(model, exit_layer=exit_layer, threshold=threshold)
    macs1, params1 = profile(
        encoder, inputs=(images_list,), custom_ops=custom_ops, verbose=True
    )
    macs, params = clever_format([macs1, params1], "%.3f")
    print('encoder macs is ', macs)
    print('encoder params is ', params)
    
    '''Speed Test'''
    T_w = 500
    T_t = 2000
    print("testing speed ...")
    with torch.no_grad():
        for i in range(T_w):
            _ = model.inference_encoder(images_list, exit_layer, threshold)
        start = time.time()
        for i in range(T_t):
            _ = model.inference_encoder(images_list, exit_layer, threshold)
        end = time.time()
        avg_lat = (end - start) / (T_t * bs)
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))


def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    return img_patch

if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)
    # Compute the Flops and Params of our STARK-S model
    args = parse_args()
    '''update cfg'''
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    bins = cfg.MODEL.BINS
    start = bins + 1
    input_start = (torch.ones([bs, 1]).to(device) * start).type(dtype=torch.int64)
    seq_list = [input_start]
    input_seqs = torch.cat(seq_list, dim=1)
    if cfg.TEST.WINDOW == True:
        hanning = torch.tensor(np.hanning(bins)).unsqueeze(0).cuda()
    else:
        hanning = None
    '''import fasterseqtrack network module'''
    model_module = importlib.import_module('lib.models.fasterseqtrack')
    model_constructor = model_module.build_fasterseqtrack
    model = model_constructor(cfg)
    # get the template and search
    template = get_data(bs, z_sz)
    search = get_data(bs, x_sz)
    # transfer to device
    model = model.to(device)
    template = template.to(device)
    search = search.to(device)
    model.eval()
    # evaluate the model properties
    images_list = [template, template, search]
    exit_layer = args.exit_layer
    threshold = args.threshold
    xz, _, _, _ = model.inference_encoder(images_list, exit_layer, threshold)
    evaluate(model, images_list, xz, input_seqs, hanning, bs=bs,
             exit_layer=exit_layer, threshold=threshold)

