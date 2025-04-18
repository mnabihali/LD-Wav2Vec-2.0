import torch
import numpy as np
import wandb
import sys
import os
import argparse
from distutils.util import strtobool
from pprint import pprint
import evaluate
#from datasets import load_metric
from modeling_wav2vec import Wav2Vec2ForCTC  # Original WavLMModel
from utils import (
    LibriSpeechDataset,
    DataCollatorCTCWithPadding,
    train_model,
    fix_seed
)
sys.path.append(os.pardir)

# from utils_ted import (
#    TEDSpeechDataset,
#    DataCollatorCTCWithPadding,
# )
import warnings
warnings.filterwarnings("ignore")

fix_seed(42)

def main():
    wandb.init()
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str, default='sample_run')
    parser.add_argument('--use_skip', type=strtobool, default=False)
    parser.add_argument('--use_adapter_fc', type=strtobool, default=False)
    parser.add_argument('--use_adapter_norm', type=strtobool, default=False)
    parser.add_argument('--eadapter_act', default='gelu')
    parser.add_argument('--ladapter_act', default='gelu')
    parser.add_argument('--lada_emb_size', type=int, default=512)
    parser.add_argument('--eada_emb_size', type=int, default=256)
    parser.add_argument('--train_encada', type=strtobool, default=False)
    parser.add_argument('--train_eadapter', type=strtobool, default=False)
    parser.add_argument('--use_adapter_ff', type=strtobool, default=False)
    parser.add_argument('--use_adapter_attn', type=strtobool, default=False)
    parser.add_argument('--adapter_init_std', type=float, default=1e-3)
    parser.add_argument('--ladapter_init_std', type=float, default=1e-3)

    parser.add_argument('--classifier_lr', type=float, default=1e-3)
    parser.add_argument('--encoder_lr', type=float, default=1e-4)
    parser.add_argument('--ladapter_lr', type=float, default=1e-3)
    parser.add_argument('--eadapter_lr', type=float, default=1e-3)

    parser.add_argument('--train_encoder', type=strtobool, default=False)
    parser.add_argument('--weighted_sum', type=strtobool, default=False)
    parser.add_argument('--train_lawithea', type=strtobool, default=False)
    parser.add_argument('--use_steplr', type=strtobool, default=True)

    parser.add_argument('--wandb_log', type=strtobool, default=False)
    parser.add_argument('--save_model', type=strtobool, default=True)

    args = parser.parse_args()

    #train_dataset = LibriSpeechDataset('librispeech_dataset_train.csv')
    train_dataset = LibriSpeechDataset('librispeech_dataset_train_all.csv')
    # val_dataset = LibriSpeechDataset('librispeech_dataset_test.csv')
    val_dataset = LibriSpeechDataset('librispeech_dataset_dev.csv')

    # train_dataset = TEDSpeechDataset('ted_train.csv')
    # val_dataset = TEDSpeechDataset('tt_ted.csv')

    processor = train_dataset.processor
    collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    if args.train_encoder:
        model_config = {'ctc_loss_reduction': 'mean',
                        'ctc_zero_infinity': True,
                        'pad_token_id': processor.tokenizer.pad_token_id,
                        }
        learning_rate = {
            'classifier': args.classifier_lr,
            'encoder': args.encoder_lr,
        }

    elif args.weighted_sum:
        model_config = {'ctc_loss_reduction': 'mean',
                        'ctc_zero_infinity': True,
                        'pad_token_id': processor.tokenizer.pad_token_id,
                        'use_adapter_to_output': True,
                        'adapter_to_output_layer_size': {str(i): args.lada_emb_size for i in range(0, 12)},
                        'use_adapter_to_output_weighted_sum': True,
                        'use_adapter_fc': False,
                        'use_upsampling': False,
                        'use_residual': False,
                        'ladapter_act': None,
                        'use_adapter_norm': False,
                        'use_weighted_layer_sum': True,
                        }
        learning_rate = {
            'classifier': args.classifier_lr,
            'adapter_layer_weights': args.ladapter_lr,
            'layer_norm': args.ladapter_lr,
        }

    elif args.train_encada:
        model_config = {'ctc_loss_reduction': 'mean',
                        'ctc_zero_infinity': True,
                        'pad_token_id': processor.tokenizer.pad_token_id,
                        'adapter_embedding_size': {str(i): args.eada_emb_size for i in range(0, 12)},
                        'eadapter_act': None if args.eadapter_act == 'None' else args.eadapter_act,
                        'use_adapter_ff': args.use_adapter_ff,
                        'use_adapter_attn': args.use_adapter_attn,
                        'adapter_init_std': args.adapter_init_std
                        }
        learning_rate = {
            'classifier': args.classifier_lr,
            'adapter_ff': args.eadapter_lr,
            'adapter_attn': args.eadapter_lr,
            'layer_norm': args.eadapter_lr,
        }

    elif args.train_lawithea:
        model_config = {'ctc_loss_reduction': 'mean',
                        'ctc_zero_infinity': True,
                        'pad_token_id': processor.tokenizer.pad_token_id,
                        'use_adapter_to_output': True,
                        'adapter_to_output_layer_size': {str(i): args.lada_emb_size for i in range(0, 12)},
                        'use_adapter_fc': args.use_adapter_fc,
                        'use_upsampling': args.use_skip,
                        'use_residual': args.use_skip,
                        'use_adapter_norm': args.use_adapter_norm,
                        'adapter_embedding_size': {str(i): args.eada_emb_size for i in range(0, 11)},
                        'ladapter_act': None if args.ladapter_act == 'None' else args.ladapter_act,
                        'eadapter_act': None if args.eadapter_act == 'None' else args.eadapter_act,
                        'use_adapter_ff': True,
                        'use_adapter_attn': False,
                        'adapter_init_std': args.adapter_init_std
                        }
        learning_rate = {
            'classifier': args.classifier_lr,
            'adapter_to_output': args.ladapter_lr,
            'adapter_layer_weights': args.ladapter_lr,
            'adapter_ff': args.eadapter_lr,
            'layer_norm': args.eadapter_lr,
        }
    else:
        model_config = {'ctc_loss_reduction': 'mean',
                        'ctc_zero_infinity': True,
                        'pad_token_id': processor.tokenizer.pad_token_id,
                        'use_adapter_to_output': True,
                        'adapter_to_output_layer_size': {str(i): args.lada_emb_size for i in range(0, 12)},
                        'use_adapter_to_output_weighted_sum': True,
                        'use_adapter_fc': args.use_adapter_fc,
                        'use_upsampling': args.use_skip,
                        'use_residual': args.use_skip,
                        'ladapter_act': None if args.ladapter_act == 'None' else args.ladapter_act,
                        'use_adapter_norm': args.use_adapter_norm,
                        }
        learning_rate = {
            'classifier': args.classifier_lr,
            'adapter_to_output': args.ladapter_lr,
            'adapter_layer_weights': args.ladapter_lr,
            'layer_norm': args.ladapter_lr,
        }

    config = {
        "pretrained_model": 'facebook/wav2vec2-base',
        "dataset": 'LL10h',
        "epochs": 500,
        "batch_size": {'train': 64, 'val': 4},
        "model_config": model_config,
        "learning_rate": learning_rate,
        'optimizer': 'Adam',
        "scheduler": {'type': 'StepLR', 'step': 25, 'gamma': 0.3} if args.train_encada and args.use_steplr else {
            'type': 'LambdaLR', 'param': {'alpha': 0.20, 'beta': 0.03, 'start': 10, 'end': 1.0, 'scale': 10}},
    }

    if args.wandb_log:
        wandb.init(
            project="ASR",
            config=config,
            id=args.run_name
        )

    num_epochs = config['epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    sc_setting = config['scheduler']
    pretrained_model = config['pretrained_model']

    ####################################################################################################################################
    if args.train_encoder:
        #PATH = "/stek/mohamed/ld_wav2vec2/che/model_epoch_170.pth"  ### To load the checkpoint to resume training
        # model = WavLMForCTC.from_pretrained(pretrained_model, ignore_mismatched_sizes=True, **model_config)
        model = Wav2Vec2ForCTC.from_pretrained(pretrained_model, **model_config)
        #model.load_state_dict(torch.load(PATH))  ### To load the checkpoint to resume training
        print("Using Wav2Vec2ForCTC model")
    else:
        print('No model founds')

    for params in model.parameters():
        params.requires_grad = True

    ####################################################################################################################################

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

    if args.train_encoder:
        layer_names = [str(i) for i in range(0, 12)]
    elif args.weighted_sum:
        layer_names = [str(i) for i in range(12)]
    elif args.train_encada:
        layer_names = ['layers.' + k for k in model_config["adapter_embedding_size"].keys()]
    else:
        layer_names = ['layers.' + k for k in model_config["adapter_to_output_layer_size"].keys()]

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

    config.update({'num_params (1e7)': pcount / 1e7})
    config.update({'num_adapter_params (M)': adapcount / 1e6})

    if args.train_encoder:
        optimizer = torch.optim.Adam([
            {'params': down_param, 'lr': learning_rate['classifier']},
            {'params': encoder_param, 'lr': learning_rate['encoder']},
        ])

    elif args.weighted_sum:
        optimizer = torch.optim.Adam([
            {'params': down_param, 'lr': learning_rate['classifier']},
            {'params': adapter_to_output_layer_weights_param, 'lr': learning_rate['adapter_layer_weights']},
            {'params': layernorm_param, 'lr': learning_rate['layer_norm']},
        ])

    elif args.train_encada:
        optimizer = torch.optim.Adam([
            {'params': down_param, 'lr': learning_rate['classifier']},
            {'params': adapter_ff_param, 'lr': learning_rate['adapter_ff']},
            {'params': adapter_attn_param, 'lr': learning_rate['adapter_attn']},
            {'params': layernorm_param, 'lr': learning_rate['layer_norm']},
        ])

    elif args.train_lawithea:
        optimizer = torch.optim.Adam([
            {'params': down_param, 'lr': learning_rate['classifier']},
            {'params': adapter_ff_param, 'lr': learning_rate['adapter_ff']},
            {'params': adapter_to_output_layer_weights_param, 'lr': learning_rate['adapter_layer_weights']},
            {'params': adapter_to_output_param, 'lr': learning_rate['adapter_to_output']},
            {'params': layernorm_param, 'lr': learning_rate['layer_norm']},
        ])

    else:
        optimizer = torch.optim.Adam([
            {'params': down_param, 'lr': learning_rate['classifier']},
            {'params': adapter_to_output_layer_weights_param, 'lr': learning_rate['adapter_layer_weights']},
            {'params': adapter_to_output_param, 'lr': learning_rate['adapter_to_output']},
            {'params': layernorm_param, 'lr': learning_rate['layer_norm']},
        ])

    if args.train_encada and args.use_steplr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sc_setting['step'], gamma=sc_setting['gamma'])
    else:
        hyparam = sc_setting['param']

        def func(epoch):
            alpha = hyparam['alpha'];
            beta = hyparam['beta']
            start = hyparam['start'];
            end = hyparam['end'];
            scale = hyparam['scale']
            warmup = np.linspace(start, num_epochs, int(num_epochs * alpha)) / num_epochs
            stag = np.ones(int(num_epochs * (beta)))
            decay = np.linspace(num_epochs, end, int(num_epochs * (1 - alpha - beta) + 1)) / np.linspace(num_epochs,
                                                                                                         num_epochs * scale,
                                                                                                         int(num_epochs * (
                                                                                                                     1 - alpha - beta) + 1))
            steps = np.concatenate([warmup, stag, decay], axis=-1)
            # print('Steps is ...........', steps)
            # print('len of steps......', len(steps))
            return steps[epoch - 1]

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func)

    metric = evaluate.load('wer', trust_remote_code=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size['train'], collate_fn=collator,
                                               shuffle=True, num_workers=12, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size['val'], collate_fn=collator,
                                             shuffle=False, num_workers=12, pin_memory=True)
    dataloaders_dict = {'train': train_loader, 'val': val_loader}

    # model.load_state_dict(torch.load(args.run_name+'/epoch-000.pth'))
    # model = train_model(model, processor, dataloaders_dict, optimizer, scheduler, metric, num_epochs, args.run_name,args.train_encoder,args.train_lawithea, args.EEloss,report_wandb=True, val_interval=100)
    model = train_model(model, processor, dataloaders_dict, optimizer, scheduler, metric, num_epochs, log_interval=5, report_wandb=False, val_interval=5, save_interval=1, save_dir='./che_again')

    print("TRAIN ENDED!")
    if args.save_model:
        torch.save(model.state_dict(), args.run_name + '/final.pth')



if __name__ == '__main__':
    main()
