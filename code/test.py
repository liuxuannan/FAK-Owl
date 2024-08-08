import os
from model.openllama import OpenLLAMAPEFTModel
from model.ImageBind.data import  load_and_transform_news_text,load_choice_data
import torch
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from PIL import Image
import numpy as np
import argparse
import yaml
from datasets import create_dataset, create_loader
import json
import torch.nn.functional as F
from transformers import GenerationConfig

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import datetime

parser = argparse.ArgumentParser("FKA_Owl", add_help=True)
# paths
parser.add_argument("--FKA_Owl_ckpt_path", default='./ckpt/train_DGM4/pytorch_model.pt')
parser.add_argument("--config", default='./fake_config/test_guardian.yaml')

command_args = parser.parse_args()
time1 = datetime.datetime.now()
# init the model
args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0/',
    'delta_ckpt_path': '../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
    'stage': 2,
    'max_tgt_len': 128,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'device': 'cuda'
}

model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
delta_ckpt = torch.load(command_args.FKA_Owl_ckpt_path, map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.eval().half().cuda()

print(f'[!] init the 7b model over ...')

def predict(
    input,
    images,
    news_texts,
    max_length,
    top_p,
    temperature,
    history,
    modality_cache,
):
    generation_config = GenerationConfig.from_model_config(model.llama_model.config)
    generation_config.return_dict_in_generate = True
    generation_config.output_scores = True
    generation_config.top_p = None
    generation_config.top_k = None

    prompt_text = ''
    for idx, (q, a) in enumerate(history):
        if idx == 0:
            prompt_text += f'{q}\n### Assistant: {a}\n###'
        else:
            prompt_text += f' Human: {q}\n### Assistant: {a}\n###'
    if len(history) == 0:
        prompt_text += f'{input}'
    else:
        prompt_text += f' Human: {input}'

    response, scores = model.generate_logits({
        'prompt': prompt_text,
        'image_paths': images if images else [],
        'audio_paths': [],
        'video_paths': [],
        'thermal_paths': [],
        'news_text':news_texts if news_texts else [],
        'top_p': None,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': modality_cache
    },
    generation_config=generation_config)

    return response, scores

config = yaml.load(open(command_args.config, 'r'), Loader=yaml.Loader)
val_dataset = create_dataset(config,is_train=False)
samplers = [None]
val_loader = create_loader([val_dataset],
                               samplers,
                               batch_size=[config['batch_size_val']],
                               num_workers=[4],
                               is_trains=[False],
                               collate_fns=[val_dataset.collate])[0]


cls_nums_all = 0
cls_acc_all = 0
y_true, y_pred =[], []
device = torch.device(args['device'])

choices = ["B", "A"]
choice_ids = [model.llama_tokenizer(choice).input_ids[1] for choice in choices]

bs = 1
batch_choice_logits = torch.zeros([bs,2])
batch_label_class = []
cls_label = torch.ones(bs, dtype=torch.long).to(device)

for i, batch in enumerate(val_loader):

    for conversation in batch['texts']:
        prompts = (conversation[0]['value'])

    images = batch['images']
    label = batch['class_names'][0]
    text = batch['captions']

    resp, scores = predict(prompts, images, text, 512, 0.1, 1.0, [], [])

    choise_scores = scores[0]
    choice_logits = choise_scores[:,choice_ids]

    k = i%bs
    batch_choice_logits[k,:] = choice_logits
    batch_label_class.append(label)

    if (i+1)%bs == 0:
        choice_probs= F.softmax(batch_choice_logits,dim=1)
        logits_real_fake = torch.zeros([bs,2])
        logits_real_fake[:,0] = choice_probs[:,0]

        fake_prob = torch.sum(choice_probs[:,1:],dim = 1)
        logits_real_fake[:,1] = fake_prob

        cls_label = torch.ones(bs, dtype=torch.long)
        real_label_pos = np.where(np.array(batch_label_class) == 'orig')[0].tolist()
        cls_label[real_label_pos] = 0

        y_pred.extend(fake_prob.detach().cpu().flatten().tolist())
        y_true.extend(cls_label.cpu().flatten().tolist())

        pred_acc = logits_real_fake.argmax(1).to(device)
        cls_nums_all += cls_label.shape[0]
        cls_label = cls_label.to(device)
        cls_acc_all += torch.sum(pred_acc == cls_label).item()

        del batch_label_class
        batch_label_class = []


y_true, y_pred = np.array(y_true), np.array(y_pred)
AUC_cls = roc_auc_score(y_true, y_pred)
ACC_cls = cls_acc_all / cls_nums_all
fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
EER_cls = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)


print("AUC:",AUC_cls)
print("ACC:",ACC_cls)
print("EER:",EER_cls)
print(cls_acc_all)
print(cls_nums_all)
time2 = datetime.datetime.now()
print("time consumed: ", time2 - time1)


