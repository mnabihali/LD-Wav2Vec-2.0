from modeling_wav2vec import Wav2Vec2ForCTC
import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser()

parser.add_argument('--run_name', type=str, default='sample_run')
parser.add_argument('--use_skip', type=strtobool, default=False)
parser.add_argument('--use_adapter_fc', type=strtobool, default=True)
parser.add_argument('--use_adapter_norm', type=strtobool, default=True)
parser.add_argument('--eadapter_act', default='gelu')
parser.add_argument('--ladapter_act', default='gelu')
parser.add_argument('--lada_emb_size', type=int, default=512)
parser.add_argument('--eada_emb_size', type=int, default=256)
parser.add_argument('--train_encada', type=strtobool, default=False)
parser.add_argument('--train_eadapter', type=strtobool, default=False)
parser.add_argument('--use_adapter_ff', type=strtobool, default=True)
parser.add_argument('--use_adapter_attn', type=strtobool, default=True)
parser.add_argument('--adapter_init_std', type=float, default=1e-3)
parser.add_argument('--ladapter_init_std', type=float, default=1e-3)

parser.add_argument('--classifier_lr', type=float, default=1e-3)
parser.add_argument('--encoder_lr', type=float, default=1e-4)
parser.add_argument('--ladapter_lr', type=float, default=1e-3)
parser.add_argument('--eadapter_lr', type=float, default=1e-3)

parser.add_argument('--train_encoder', type=strtobool, default=False)
parser.add_argument('--weighted_sum', type=strtobool, default=False)
parser.add_argument('--train_lawithea', type=strtobool, default=False)

parser.add_argument('--wandb_log', type=strtobool, default=False)

args = parser.parse_args()

model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")
layer_names = [str(i) for i in range(0, 12)]

down_param = []
layernorm_param = []
encoder_param = []
adapter_ff_param = []
adapter_attn_param = []
adapter_to_output_param = []
adapter_to_output_layer_weights_param = []
pcount = 0
adapcount = 0
flag = True

for name, param in model.named_parameters():
    for layer in layer_names:
        if layer in name:
            flag = True
            break
        else:
            flag = False

    if 'lm_head' in name:
        print('down_param: ', name)
        pcount += param.numel()
        down_param.append(param)

    elif 'adapter_to_output_layer_weights' in name:
        adapter_to_output_layer_weights_param.append(param)
        print('adapter_to_output_layer_weights: ', name)
        pcount += param.numel();
        adapcount += param.numel()

    elif 'encoder.layers' in name and 'layer_norm' in name and flag and not args.train_encoder:
        layernorm_param.append(param)
        print('layer_norm: ', name);
        pcount += param.numel()

    elif 'adapter_layer_ff' in name:
        adapter_ff_param.append(param)
        print('enc_adapter_ff: ', name)
        pcount += param.numel();
        adapcount += param.numel()

    elif 'adapter_layer_attn' in name:
        adapter_attn_param.append(param)
        print('enc_adapter_attn: ', name)
        pcount += param.numel();
        adapcount += param.numel()

    elif 'adapter_to_output' in name:
        adapter_to_output_param.append(param)
        print('adapter_output: ', name)
        pcount += param.numel();
        adapcount += param.numel()

    elif 'encoder.layers' in name and flag and args.train_encoder:
        encoder_param.append(param)
        pcount += param.numel();
        print('encoder: ', name)

    elif 'wav2vec2.policy_model' in name:
        print('Un-freezing the Policy Module, --', name)
        param.requires_grad = True

    else:
        # print('Not freezing: ', name)
        print('frozen: ', name)
        pcount += param.numel()
        param.requires_grad = False
        # param.requires_grad = True

print('tot. number of parameters: ', pcount, 'tot. number of adapters parameters: ', adapcount, '[',
      (adapcount / pcount) * 100, "%]")