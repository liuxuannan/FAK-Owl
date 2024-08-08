from header import *
import torch.nn.functional as F
from .ImageBind import *
from .ImageBind import data
from .modeling_llama import LlamaForCausalLM
from .FKA_Owl_models import LinearLayer, Cross_Modal_Reason, Segmentation_Verification, Bbox_Verification
from transformers import StoppingCriteria, StoppingCriteriaList
from utils.loss import FocalLoss, BinaryDiceLoss
import kornia as K
from peft import LoraConfig, TaskType, get_peft_model
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig, GenerationConfig
import torch
from torch.nn.utils import rnn
from model import box_ops
from utils.multilabel_metrics import get_multi_label

CLASS_NAMES = ['face', 'object']

prompt_normal = ['{}', 'genuine {}', 'natural {}', 'realistic {}', '{} without blending boundaries', '{} without inconsistent textures', '{} without unnatural shadows']
prompt_abnormal = ['synthetic {}', 'unnatural {}', 'unrealistic {}','{} with blending boundaries', '{} with inconsistent textures', '{} with unnatural shadows']


prompt_state = [prompt_normal, prompt_abnormal]
prompt_templates = ['a photo of a {}.', 'a photo of the {}.']
objs = ['face', 'object']

prompt_sentences = {}

for obj in objs:
    prompt_sentence_obj = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(obj) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = data.load_and_transform_text(prompted_sentence, torch.cuda.current_device())
        prompt_sentence_obj.append(prompted_sentence)
    prompt_sentences[obj] = prompt_sentence_obj

def encode_text(model, text, fake_text_pos,device):

    text_sentence,padding_masks,fake_token_pos_batch = data.load_and_transform_news_text(text, fake_text_pos,device)

    inputs = {ModalityType.TEXT: text_sentence}

    text_features = model(inputs)[ModalityType.TEXT][0]
    text_patch_feaures = model(inputs)[ModalityType.TEXT][1]
    text_patch_feaure = text_patch_feaures[2].permute(1,0,2)

    return text_features.unsqueeze(1), text_patch_feaure, padding_masks, fake_token_pos_batch

def encode_text_reference(model, text, device):

    text_sentence,padding_masks = data.load_and_transform_news_text_reference(text, device)

    inputs = {ModalityType.TEXT: text_sentence}

    text_features = model(inputs)[ModalityType.TEXT][0]
    text_patch_feaures = model(inputs)[ModalityType.TEXT][1]
    text_patch_feaure = text_patch_feaures[2].permute(1,0,2)

    return text_features.unsqueeze(1), text_patch_feaure, padding_masks


def encode_text_with_prompt_ensemble(model, obj, device):

    global prompt_sentences
    normal_sentences = []
    abnormal_sentences = []
    for idx in range(len(obj)):
        sentence = prompt_sentences[obj[idx].replace('_', ' ')]
        normal_sentences.append(sentence[0])
        abnormal_sentences.append(sentence[1])

    normal_sentences = torch.cat(normal_sentences).to(device)
    abnormal_sentences = torch.cat(abnormal_sentences).to(device)

    class_embeddings_normal = model({ModalityType.TEXT: normal_sentences})[ModalityType.TEXT][0]
    class_embeddings_abnormal = model({ModalityType.TEXT: abnormal_sentences})[ModalityType.TEXT][0]
    # class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

    class_embeddings_normal = class_embeddings_normal.reshape((len(obj), len(prompt_templates) * len(prompt_normal), 1024))
    class_embeddings_normal = class_embeddings_normal.mean(dim=1, keepdim=True)
    class_embeddings_normal = class_embeddings_normal / class_embeddings_normal.norm(dim=-1, keepdim=True)

    class_embeddings_abnormal = class_embeddings_abnormal.reshape((len(obj), len(prompt_templates) * len(prompt_abnormal), 1024))
    class_embeddings_abnormal = class_embeddings_abnormal.mean(dim=1, keepdim=True)
    class_embeddings_abnormal = class_embeddings_abnormal / class_embeddings_abnormal.norm(dim=-1, keepdim=True)

    text_features = torch.cat([class_embeddings_normal, class_embeddings_abnormal], dim=1)

    return text_features



class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()
        if stop_count >= self.ENCOUNTERS:
            return True
        return False

def build_one_instance(tokenizer, conversation):
    text_list = []
    turn_num = len(conversation)
    input_ids, target_ids = [], []
    for i in range(turn_num):
        turn = conversation[i]
        role = turn['from']
        if i == 0: # the first human turn
            assert role == 'human'
            text = turn['value'] + '\n### Assistant:'
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100]*len(one_input_id) # do not perform loss regression on human prompt
        else:
            if role == 'human':
                text = 'Human: ' + turn['value'] + '\n### Assistant:'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += [-100]*len(one_input_id)
            elif role == 'gpt':
                text = turn['value'] + '\n###'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
            else:
                raise Exception('Wrong Role!!!')
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    return text_list, input_ids, target_ids

def process_batch_instance(tokenizer, batch_of_conversations, max_tgt_len):
    batch_input_ids, batch_target_ids = [], []
    for conversation in batch_of_conversations:
        _, one_input_ids, one_target_ids = build_one_instance(tokenizer, conversation)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:,:max_tgt_len]
    target_ids = target_ids[:,:max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()

def find_first_file_in_directory(directory_path):
    try:
        file_list = os.listdir(directory_path)
        for item in file_list:
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                return item_path
        return None

    except OSError as e:
        print(f"Error while accessing directory: {e}")
        return None


PROMPT_START = '### Human: <Img>'
class OpenLLAMAPEFTModel(nn.Module):

    '''LoRA for LLaMa model'''

    def __init__(self, **args):
        super(OpenLLAMAPEFTModel, self).__init__()
        self.args = args
        imagebind_ckpt_path = args['imagebind_ckpt_path']
        vicuna_ckpt_path = args['vicuna_ckpt_path']
        max_tgt_len = args['max_tgt_len']
        stage = args['stage']

        print (f'Initializing visual encoder from {imagebind_ckpt_path} ...')

        self.visual_encoder, self.visual_hidden_size = imagebind_model.imagebind_huge(args)
        imagebind_ckpt = torch.load(imagebind_ckpt_path, map_location=torch.device('cpu'))
        self.visual_encoder.load_state_dict(imagebind_ckpt, strict=True)

        self.iter = 0

        self.Cross_Modal_Reason = Cross_Modal_Reason(1024)
        self.Multi_level = LinearLayer(1280, 1024, 4)
        
        self.Segmentation_Verification = Segmentation_Verification(1, 4096)
        self.Bbox_Verification = Bbox_Verification(1024)

        self.loss_focal = FocalLoss()
        self.loss_dice = BinaryDiceLoss()

        # free vision encoder
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
        print ('Visual encoder initialized.')

        print (f'Initializing language decoder from {vicuna_ckpt_path} ...')
        
        # add the lora module
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=self.args['lora_r'], 
            lora_alpha=self.args['lora_alpha'], 
            lora_dropout=self.args['lora_dropout'],
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
        )

        self.llama_model = LlamaForCausalLM.from_pretrained(vicuna_ckpt_path)
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()

        self.llama_tokenizer = LlamaTokenizer.from_pretrained(vicuna_ckpt_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        print ('Language decoder initialized.')

        self.llama_proj = nn.Linear(
            self.visual_hidden_size, self.llama_model.config.hidden_size
        )

        self.max_tgt_len = max_tgt_len
        self.device = torch.cuda.current_device()

    def get_bbox_loss(self, output_coord, target_bbox, is_image=None):
        """
        Bounding Box Loss: L1 & GIoU

        Args:
            image_embeds: encoding full images
        """
        loss_bbox = F.l1_loss(output_coord, target_bbox, reduction='none')  # bsz, 4

        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(target_bbox)
        if (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any():
            # early check of degenerated boxes
            print("### (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any()")
            loss_giou = torch.zeros(output_coord.size(0), device=output_coord.device)
        else:
            # loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(boxes1, boxes2))  # bsz
            loss_giou = 1 - box_ops.generalized_box_iou(boxes1, boxes2)  # bsz

        if is_image is None:
            num_boxes = target_bbox.size(0)
        else:
            num_boxes = torch.sum(1 - is_image)
            loss_bbox = loss_bbox * (1 - is_image.view(-1, 1))
            loss_giou = loss_giou * (1 - is_image)

        return loss_bbox.sum() / num_boxes, loss_giou.sum() / num_boxes


    def rot90_img(self,x,k):
        # k is 0,1,2,3
        degreesarr = [0., 90., 180., 270., 360]
        degrees = torch.tensor(degreesarr[k]).to(self.llama_model.dtype).to(self.device)
        x = K.geometry.transform.rotate(x, angle = degrees, padding_mode='reflection')
        return x

    def encode_video(self, video_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_video_data(video_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            video_embeds = embeddings[ModalityType.VISION][0] # bsz x 1024
        inputs_llama = self.llama_proj(video_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama

    def encode_audio(self, audio_paths):
        inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            audio_embeds = embeddings[ModalityType.AUDIO][0] # bsz x 1024
        inputs_llama = self.llama_proj(audio_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama

    def encode_thermal(self, thermal_paths):
        inputs = {ModalityType.THERMAL: data.load_and_transform_thermal_data(thermal_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['thermal'][0] # bsz x 1024
        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama

    def encode_image(self, image_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0] # bsz x 1024
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
        patch_tokens = self.Multi_level(patch_features) # bsz x h*w x 1024

        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama, patch_tokens
    
    def encode_image_for_web_demo(self, image_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data_for_web_demo(image_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0] # bsz x 1024
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
        patch_tokens = self.Multi_level(patch_features) # bsz x h*w x 1024

        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama, patch_tokens
    
    def encode_image_for_one_shot(self, image_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
            for i in range(len(patch_features)):
                patch_features[i] = patch_features[i].transpose(0, 1)[:, 1:, :]

        return patch_features
    
    def encode_image_for_one_shot_from_tensor(self, image_tensors):
        if not isinstance(image_tensors, list):
            image_tensors = [image_tensors]
        inputs = {ModalityType.VISION: torch.stack(image_tensors, dim=0).to(self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
            for i in range(len(patch_features)):
                patch_features[i] = patch_features[i].transpose(0, 1)[:, 1:, :]

        return patch_features
    
    def encode_image_for_one_shot_with_aug(self, image_paths):
        image_tensors = data.load_and_transform_vision_data(image_paths, self.device).to(self.llama_model.dtype)
        B,C,H,W = image_tensors.shape
        # print(B,C,H,W)

        rotated_images = torch.zeros((4, B, C, H, W)).to(self.llama_model.dtype).to(self.device)


        for j, degree in enumerate([0, 1, 2, 3]):
            rotated_img = self.rot90_img(image_tensors, degree)
            # 存储旋转后的图像
            rotated_images[j] = rotated_img

        image_tensors = rotated_images.transpose(0,1).reshape(B * 4, C, H, W)

        inputs = {ModalityType.VISION: image_tensors}
        # convert into visual dtype
        inputs = {key: inputs[key] for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
            for i in range(len(patch_features)):
                patch_features[i] = patch_features[i].transpose(0, 1)[:, 1:, :].reshape(B,4,256,1280).reshape(B, 4 * 256, 1280)

        return patch_features
    
    def encode_image_from_tensor(self, image_tensors):
        if not isinstance(image_tensors, list):
            image_tensors = [image_tensors]
        inputs = {ModalityType.VISION: torch.stack(image_tensors, dim=0).to(self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0] # bsz x 1024
            patch_features = embeddings['vision'][1] # bsz x h*w x 1024

        patch_tokens = self.Multi_level(patch_features)

        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama, patch_tokens, image_embeds.unsqueeze(1)

    



    def prompt_wrap(self, img_embeds, input_ids, target_ids, attention_mask, forgery_embedding = None):
        '''
            input_ids, target_ids, attention_mask: bsz x s2
        '''
        input_ids = input_ids.to(self.device) # bsz x s2
        target_ids = target_ids.to(self.device) # bsz x s2
        attention_mask = attention_mask.to(self.device) # bsz x s2

        batch_size = img_embeds.shape[0]
        p_before = PROMPT_START
        p_before_tokens = self.llama_tokenizer(p_before, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        # peft model need deeper call
        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim

        p_middle = '</Img> '
        #p_middle = 'According to multimodal content consistency information  </Img> '
        p_middle_tokens = self.llama_tokenizer(p_middle, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        # peft model need deeper call
        p_middle_embeds = self.llama_model.model.model.embed_tokens(p_middle_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim


        p_after_embeds = self.llama_model.model.model.embed_tokens(input_ids).expand(batch_size, -1, -1) # bsz x s2 x embed_dim
        bos = torch.ones([batch_size, 1],
                         dtype=p_before_tokens.input_ids.dtype,
                         device=p_before_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id # bsz x 1
        bos_embeds = self.llama_model.model.model.embed_tokens(bos) # bsz x 1 x embed_dim



        if forgery_embedding != None:
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds, p_middle_embeds, forgery_embedding, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
            # create targets
            empty_targets = (
                torch.ones([batch_size, 1+p_before_embeds.size()[1]+1+p_middle_embeds.size()[1] + forgery_embedding.size()[1]], # 1 (bos) + s1 + 1 (image vector)
                        dtype=torch.long).to(self.device).fill_(-100)  
            ) # bsz x (1 + s1 + 1)
            targets = torch.cat([empty_targets, target_ids], dim=1) # bsz x (1 + s1 + 1 + s2)
            assert inputs_embeds.size()[1] == targets.size()[1]

            atts_prefix = torch.ones([batch_size, 1+p_before_embeds.size()[1]+1+p_middle_embeds.size()[1] + forgery_embedding.size()[1]], dtype=torch.long).to(self.device) # bsz x (1 + s1 +1)
            attention_mask = torch.cat([atts_prefix, attention_mask], dim=1)
            assert attention_mask.size() == targets.size() # bsz x (1 + s1 + 1 + s2)
            return inputs_embeds, targets, attention_mask 
        else:
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds, p_middle_embeds, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
            # create targets
            empty_targets = (
                torch.ones([batch_size, 1+p_before_embeds.size()[1]+1+p_middle_embeds.size()[1]], # 1 (bos) + s1 + 1 (image vector)
                        dtype=torch.long).to(self.device).fill_(-100)  
            ) # bsz x (1 + s1 + 1)
            targets = torch.cat([empty_targets, target_ids], dim=1) # bsz x (1 + s1 + 1 + s2)
            assert inputs_embeds.size()[1] == targets.size()[1]

            atts_prefix = torch.ones([batch_size, 1+p_before_embeds.size()[1]+1+p_middle_embeds.size()[1]], dtype=torch.long).to(self.device) # bsz x (1 + s1 +1)
            attention_mask = torch.cat([atts_prefix, attention_mask], dim=1)
            assert attention_mask.size() == targets.size() # bsz x (1 + s1 + 1 + s2)
            return inputs_embeds, targets, attention_mask 


    def forward(self, inputs):
            
        # Obtain multi_level visual feauture
        image_paths = inputs['images']
        img_embeds, _, patch_tokens, img_embeds_before_proj = self.encode_image_from_tensor(image_paths)
        img_patch_feaure_layers = torch.stack(patch_tokens,dim=0)
        img_patch_feaure = torch.mean(img_patch_feaure_layers,dim = 0)
        img_all_feature = torch.cat([img_embeds_before_proj,img_patch_feaure],dim=1)

        text = inputs['captions']
        fake_text_pos = inputs['fake_text_pos_list']
        device = self.device
        bs = img_embeds.shape[0]

        label = inputs['class_names']
        multicls_label, real_label_pos = get_multi_label(label, device)
        itm_labels = torch.ones(bs, dtype=torch.long).to(device)
        itm_labels[real_label_pos] = 0  

        loss_pixel = 0
        feats_text_tensor = encode_text_with_prompt_ensemble(self.visual_encoder, ['object' for _ in label],
                                                                self.device)


        news_text_embeds, news_text_patch_embeds, padding_masks, fake_token_pos_batch = \
            encode_text(self.visual_encoder, text, fake_text_pos, self.device)
        text_all_feature = torch.cat([news_text_embeds, news_text_patch_embeds], dim=1)

        # cross-modal reason
        forgery_embed, forgery_patch_embeds = self.Cross_Modal_Reason(bs, img_all_feature, text_all_feature)

        # #bbox verfication
        fake_image_box = torch.cat(inputs['fake_image_boxs'], dim=0).reshape(bs, -1)
        output_coord, atts_local_feat_aggr = self.Bbox_Verification(forgery_patch_embeds)
        loss_bbox, loss_giou = self.get_bbox_loss(output_coord, fake_image_box.to(output_coord.device))

        # segmentation verification
        forgery_map_prompts, forgery_maps = self.Segmentation_Verification(forgery_patch_embeds,feats_text_tensor,atts_cls_feat = forgery_embed, atts_bbox_feat = atts_local_feat_aggr)


        gt = inputs['masks']
        gt = torch.stack(gt, dim=0).to(self.device)
        gt = gt.squeeze()
        gt[gt > 0.3], gt[gt <= 0.3] = 1, 0
        f_loss = self.loss_focal(forgery_maps, gt)
        d_loss = self.loss_dice(forgery_maps[:, 1, :, :], gt)
        loss_pixel = loss_pixel + f_loss + d_loss

        output_texts = inputs['texts']
        input_ids, target_ids, attention_mask = process_batch_instance(self.llama_tokenizer, output_texts, self.max_tgt_len)
        inputs_embeds, targets, attention_mask = self.prompt_wrap(img_embeds, input_ids, target_ids, attention_mask, forgery_map_prompts)
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss


        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]    # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)    # [B*S]
        valid_mask = (labels != -100).reshape(-1)

        valid_tokens = gen_acc & valid_mask    # [B*S]
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()

        return loss + loss_pixel+ 0.1*(loss_giou + loss_bbox), gen_acc



    def extract_multimodal_feature(self, inputs):
        features = []
        if inputs['image_paths']:

            image_embeds, _, patch_tokens, img_embeds_before_proj = self.encode_image_from_tensor(inputs['image_paths'])

            img_patch_feaure_layers = torch.stack(patch_tokens, dim=0)
            img_patch_feaure = torch.mean(img_patch_feaure_layers, dim=0)
            img_all_feature = torch.cat([img_embeds_before_proj, img_patch_feaure], dim=1)

            text = inputs['news_text']
            news_text_embeds, news_text_patch_embeds, padding_masks = encode_text_reference(self.visual_encoder, text, self.device)
            text_all_feature = torch.cat([news_text_embeds, news_text_patch_embeds], dim=1)


            features.append(image_embeds)
        if inputs['audio_paths']:
            audio_embeds, _ = self.encode_audio(inputs['audio_paths'])
            features.append(audio_embeds)
        if inputs['video_paths']:
            video_embeds, _ = self.encode_video(inputs['video_paths'])
            features.append(video_embeds)
        if inputs['thermal_paths']:
            thermal_embeds, _ = self.encode_thermal(inputs['thermal_paths'])
            features.append(thermal_embeds)

        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)

        return feature_embeds, img_all_feature, text_all_feature


    def prepare_generation_embedding(self, inputs):
        prompt = inputs['prompt']
        
        feature_embeds, news_img_embeds, news_text_embeds = self.extract_multimodal_feature(inputs)

        inputs['modality_embeds'].append(feature_embeds)

        batch_size = feature_embeds.shape[0]
        p_before = PROMPT_START
        p_before_tokens = self.llama_tokenizer(p_before, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
        
        p_middle = '</Img> '
        p_middle_tokens = self.llama_tokenizer(p_middle, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        # peft model need deeper call
        p_middle_embeds = self.llama_model.model.model.embed_tokens(p_middle_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim


        # forgery-specific knowledge augmentation
        feats_text_tensor = encode_text_with_prompt_ensemble(self.visual_encoder, ['object' for _ in range(batch_size)],
                                                             self.device)
        # consistency knowledge
        forgery_embed, forgery_patch_embeds = self.Cross_Modal_Reason(batch_size, news_img_embeds, news_text_embeds)
        
        # artifact knowledge
        output_coord, atts_local_feat_aggr = self.Bbox_Verification(forgery_patch_embeds)
        forgery_map_prompts, forgery_maps = self.Segmentation_Verification(forgery_patch_embeds, feats_text_tensor,
                                                                        atts_cls_feat=forgery_embed,
                                                                        atts_bbox_feat=atts_local_feat_aggr)
        

        text = prompt + '\n### Assistant:'
        p_after_tokens = self.llama_tokenizer(text, add_special_tokens=False, return_tensors='pt').to(self.device)
        p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s2 x embed_dim
        bos = torch.ones([batch_size, 1],
                         dtype=p_before_tokens.input_ids.dtype,
                         device=p_before_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id # bsz x 1
        bos_embeds = self.llama_model.model.model.embed_tokens(bos) # bsz x 1 x embed_dim
        inputs_embeds = torch.cat([bos_embeds, p_before_embeds, feature_embeds, p_middle_embeds, forgery_map_prompts, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
    
        return inputs_embeds

    def generate(self, inputs,web_demo=False):
        '''
            inputs = {
                'image_paths': optional,
                'audio_paths': optional
                'video_paths': optional
                'thermal_paths': optional
                'mode': generation mode,
                'prompt': human input prompt,
                'max_tgt_len': generation length,
                'top_p': top_p,
                'temperature': temperature
                'modality_embeds': None or torch.tensor
                'modality_cache': save the image cache
            }
        '''

        input_embeds = self.prepare_generation_embedding(inputs, web_demo)
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[2277], encounters=1)])

        outputs = self.llama_model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=inputs['max_tgt_len'],
            top_p=inputs['top_p'],
            temperature=inputs['temperature'],
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria
        )

        output_text = self.llama_tokenizer.decode(outputs[0][:-2], skip_special_tokens=True)
        return output_text
        
    def generate_logits(self, inputs, generation_config, web_demo=False):
        '''
            inputs = {
                'image_paths': optional,
                'audio_paths': optional
                'video_paths': optional
                'thermal_paths': optional
                'mode': generation mode,
                'prompt': human input prompt,
                'max_tgt_len': generation length,
                'top_p': top_p,
                'temperature': temperature
                'modality_embeds': None or torch.tensor
                'modality_cache': save the image cache
            }
        '''

        input_embeds = self.prepare_generation_embedding(inputs)
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[2277], encounters=1)])

        outputs_all = self.llama_model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=inputs['max_tgt_len'],
            top_p=inputs['top_p'],
            temperature=inputs['temperature'],
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config
        )
        outputs = outputs_all.sequences
        scores = outputs_all.scores
        # for i in range(len(scores)):
        #     y = torch.max(scores[i],1)
        output_text = self.llama_tokenizer.decode(outputs[0][:-2], skip_special_tokens=True)
        return output_text, scores