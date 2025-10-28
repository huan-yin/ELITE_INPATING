from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, LMSDiscreteScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from PIL import Image
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.models.clip.configuration_clip import CLIPTextConfig
from transformers.models.clip.modeling_clip import CLIP_TEXT_INPUTS_DOCSTRING, _expand_mask


import numpy as np
import torch
import albumentations as A
import cv2
import torchvision
import os
def _build_causal_attention_mask(bsz, seq_len, dtype):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
    mask.fill_(torch.tensor(torch.finfo(dtype).min))
    mask.triu_(1)  # zero out the lower diagonal
    mask = mask.unsqueeze(1)  # expand mask
    return mask

@add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPTextConfig)
def inj_forward_text(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPooling]:
    r"""
    Returns:
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is None:
        raise ValueError("You have to specify either input_ids")

    r_input_ids = input_ids['input_ids']
    if 'inj_embedding' in input_ids:
        inj_embedding = input_ids['inj_embedding']
        inj_index = input_ids['inj_index']
    else:
        inj_embedding = None
        inj_index = None

    input_shape = r_input_ids.size()
    r_input_ids = r_input_ids.view(-1, input_shape[-1])


    inputs_embeds = self.embeddings.token_embedding(r_input_ids)
    new_inputs_embeds = inputs_embeds.clone()
    if inj_embedding is not None:
        emb_length = inj_embedding.shape[1]
        for bsz, idx in enumerate(inj_index):
            lll = new_inputs_embeds[bsz, idx+emb_length:].shape[0]
            new_inputs_embeds[bsz, idx+emb_length:] = inputs_embeds[bsz, idx+1:idx+1+lll]
            new_inputs_embeds[bsz, idx:idx+emb_length] = inj_embedding[bsz]

    hidden_states = self.embeddings(input_ids=r_input_ids, position_ids=position_ids, inputs_embeds=new_inputs_embeds)

    bsz, seq_len = input_shape
    # CLIP's text model uses causal mask, prepare it here.
    # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
    causal_attention_mask = _build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
        hidden_states.device
    )
    # expand attention_mask
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = self.final_layer_norm(last_hidden_state)

    # text_embeds.shape = [batch_size, sequence_length, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=r_input_ids.device), r_input_ids.to(torch.int).argmax(dim=-1)
    ]

    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )

def inj_forward_crossattention(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
    context = encoder_hidden_states
    if context is not None:
        context_tensor = context["CONTEXT_TENSOR"]
    else:
        context_tensor = hidden_states

    batch_size, sequence_length, _ = hidden_states.shape

    query = self.to_q(hidden_states)
    if context is not None:
        key = self.to_k_global(context_tensor)
        value = self.to_v_global(context_tensor)
    else:
        key = self.to_k(context_tensor)
        value = self.to_v(context_tensor)

    dim = query.shape[-1]

    query = self.reshape_heads_to_batch_dim(query)
    key = self.reshape_heads_to_batch_dim(key)
    value = self.reshape_heads_to_batch_dim(value)

    attention_scores = torch.matmul(query, key.transpose(-1, -2))
    attention_scores = attention_scores * self.scale

    attention_probs = attention_scores.softmax(dim=-1)

    hidden_states = torch.matmul(attention_probs, value)
    hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)

    return hidden_states


class Mapper(nn.Module):
    def __init__(self,
        input_dim: int,
        output_dim: int,
    ):
        super(Mapper, self).__init__()

        for i in range(5):
            setattr(self, f'mapping_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                         nn.LayerNorm(1024),
                                         nn.LeakyReLU(),
                                         nn.Linear(1024, 1024),
                                         nn.LayerNorm(1024),
                                         nn.LeakyReLU(),
                                         nn.Linear(1024, output_dim)))

            setattr(self, f'mapping_patch_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, output_dim)))

    def forward(self, embs):
        hidden_states = ()
        for i, emb in enumerate(embs):
            hidden_state = getattr(self, f'mapping_{i}')(emb[:, :1]) + getattr(self, f'mapping_patch_{i}')(emb[:, 1:]).mean(dim=1, keepdim=True)
            hidden_states += (hidden_state, )
        hidden_states = torch.cat(hidden_states, dim=1)
        return hidden_states
    
def pww_load_tools(
    device: str = "cuda:0",
    scheduler_type=LMSDiscreteScheduler,
    mapper_model_path: Optional[str] = None,
    diffusion_model_path: Optional[str] = None,
    model_token: Optional[str] = None,
) -> Tuple[
    UNet2DConditionModel,
    CLIPTextModel,
    CLIPTokenizer,
    AutoencoderKL,
    CLIPVisionModel,
    Mapper,
    LMSDiscreteScheduler,
]:

    # 'CompVis/stable-diffusion-v1-4'
    # local_path_only = diffusion_model_path is not None
    vae = AutoencoderKL.from_pretrained(
        diffusion_model_path,
        subfolder="vae",
        use_auth_token=model_token,
        torch_dtype=torch.float16,
       #  local_files_only=local_path_only,
    )

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16,)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16,)
    image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16,)


    # Load models and create wrapper for stable diffusion
    for _module in text_encoder.modules():
        if _module.__class__.__name__ == "CLIPTextTransformer":
            _module.__class__.__call__ = inj_forward_text

    unet = UNet2DConditionModel.from_pretrained(
        diffusion_model_path,
        subfolder="unet",
        use_auth_token=model_token,
        torch_dtype=torch.float16,
    #    local_files_only=local_path_only,
    )

    mapper = Mapper(input_dim=1024, output_dim=768)

    for _name, _module in unet.named_modules():
        if _module.__class__.__name__ == "CrossAttention":
            if 'attn1' in _name: continue
            _module.__class__.__call__ = inj_forward_crossattention

            shape = _module.to_k.weight.shape
            to_k_global = nn.Linear(shape[1], shape[0], bias=False)
            mapper.add_module(f'{_name.replace(".", "_")}_to_k', to_k_global)

            shape = _module.to_v.weight.shape
            to_v_global = nn.Linear(shape[1], shape[0], bias=False)
            mapper.add_module(f'{_name.replace(".", "_")}_to_v', to_v_global)

    mapper.load_state_dict(torch.load(mapper_model_path, map_location='cpu'))
    mapper.half()

    for _name, _module in unet.named_modules():
        if 'attn1' in _name: continue
        if _module.__class__.__name__ == "CrossAttention":
            _module.add_module('to_k_global', mapper.__getattr__(f'{_name.replace(".", "_")}_to_k'))
            _module.add_module('to_v_global', mapper.__getattr__(f'{_name.replace(".", "_")}_to_v'))

    vae.to(device), unet.to(device), text_encoder.to(device), image_encoder.to(device), mapper.to(device)

    scheduler = scheduler_type(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    vae.eval()
    unet.eval()
    image_encoder.eval()
    text_encoder.eval()
    mapper.eval()
    return vae, unet, text_encoder, tokenizer, image_encoder, mapper, scheduler

def process_pil_image(image):
    if not image.mode == "RGB":
        image = image.convert("RGB")

    image_np = np.array(image)

    img = cv2.resize(image_np, (512, 512), interpolation=cv2.INTER_CUBIC)
    img = np.array(img).astype(np.float16)
    img = img / 127.5 - 1.0
    return torch.from_numpy(img).permute(2, 0, 1)


def process_pil_mask(mask):
    mask = np.array(mask.convert("L"))
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_CUBIC)
    mask = mask.astype(np.float16) / 255.0
    mask = mask[None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    return torch.from_numpy(mask)

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]
    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def th2image(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)

def update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
    """ Update the latent according to the computed loss. """
    grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
    latents = latents - step_size * grad_cond
    return latents

def compute_loss(latents, ref_image_latents, mask, ref_mask):
    """
    计算latents中mask区域与ref_image_latents中ref_mask区域的MSE损失
    
    参数:
        latents: 形状为(1,4,64,64)的张量
        ref_image_latents: 形状为(1,4,64,64)的张量
        mask: 形状为(1,1,64,64)的张量，用于提取latents中的区域
        ref_mask: 形状为(1,1,64,64)的张量，用于提取ref_image_latents中的区域
    
    返回:
        mse_loss: 两个区域调整后的MSE损失
    """
    # 处理mask，提取latents中的有效区域
    mask_squeezed = mask.squeeze(0).squeeze(0)
    non_zero_coords = torch.nonzero(mask_squeezed)
    
    if non_zero_coords.numel() == 0:
        return torch.tensor(0.0, device=latents.device)
    
    h_min, w_min = non_zero_coords.min(dim=0).values
    h_max, w_max = non_zero_coords.max(dim=0).values
    new_latents = latents[:, :, h_min:h_max+1, w_min:w_max+1]
    H_l, W_l = new_latents.shape[2], new_latents.shape[3]  # 目标区域的高和宽

    # 处理ref_mask，提取ref_image_latents中的有效区域
    ref_mask_squeezed = ref_mask.squeeze(0).squeeze(0)
    ref_non_zero_coords = torch.nonzero(ref_mask_squeezed)
    
    if ref_non_zero_coords.numel() == 0:
        return torch.tensor(0.0, device=latents.device)
    
    ref_h_min, ref_w_min = ref_non_zero_coords.min(dim=0).values
    ref_h_max, ref_w_max = ref_non_zero_coords.max(dim=0).values
    new_ref_latents = ref_image_latents[:, :, ref_h_min:ref_h_max+1, ref_w_min:ref_w_max+1]
    H_r, W_r = new_ref_latents.shape[2], new_ref_latents.shape[3]  # 参考区域的高和宽

    # 第一步：对齐宽高大小关系（宽大高小或宽小高大）
    latent_wider = W_l > H_l  # 目标区域是否宽>高
    ref_wider = W_r > H_r     # 参考区域是否宽>高
    
    if latent_wider != ref_wider:
        # 转置参考区域的宽高维度以对齐关系
        new_ref_latents = new_ref_latents.transpose(2, 3)  # 交换H和W维度
        H_r, W_r = W_r, H_r  # 更新转置后的高和宽
        ref_wider = not ref_wider  # 更新宽高关系标记

    # 第二步：计算缩放因子，使参考区域恰好包含目标区域
    scale_w = W_l / W_r  # 宽方向缩放比例
    scale_h = H_l / H_r  # 高方向缩放比例

    # 判断需要放大还是缩小参考区域
    needs_enlarge = (W_r <= W_l and H_r <= H_l) and (W_r < W_l or H_r < H_l)
    needs_shrink = (W_r >= W_l and H_r >= H_l) and (W_r > W_l or H_r > H_l)

    # 确定缩放因子（保持参考区域宽高比）
    if needs_enlarge:
        scale = max(scale_w, scale_h)  # 放大到恰好包含
    elif needs_shrink:
        scale = min(scale_w, scale_h)  # 缩小到恰好包含
    else:
        # 部分维度超出时，取最大比例确保包含
        scale = max(scale_w, scale_h)

    # 计算缩放后的尺寸（四舍五入为整数）
    scaled_H = max(1, round(H_r * scale))  # 至少为1防止无效尺寸
    scaled_W = max(1, round(W_r * scale))

    # 缩放参考区域
    resized_ref = F.interpolate(
        new_ref_latents,
        size=(scaled_H, scaled_W),
        mode='bilinear',
        align_corners=False
    )

    # 第三步：居中裁剪到与目标区域相同尺寸
    # 计算裁剪起始坐标
    h_start = max(0, (scaled_H - H_l) // 2)
    h_end = h_start + H_l
    w_start = max(0, (scaled_W - W_l) // 2)
    w_end = w_start + W_l

    # 处理边界溢出（确保裁剪区域有效）
    if h_end > scaled_H:
        h_end = scaled_H
        h_start = max(0, h_end - H_l)
    if w_end > scaled_W:
        w_end = scaled_W
        w_start = max(0, w_end - W_l)

    # 执行裁剪
    cropped_ref = resized_ref[:, :, h_start:h_end, w_start:w_end]

    # 计算MSE损失
    mse_loss = F.mse_loss(new_latents, cropped_ref)
    
    return mse_loss

def perform_iterative_refinement_step(
    latents: torch.Tensor,
    ref_latents: torch.Tensor,
    mask: torch.Tensor,
    ref_mask: torch.Tensor,
    loss: torch.Tensor,
    threshold: float,
    step_size: float,
    max_refinement_steps: int = 20,
):
    """
    Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent code
    according to our loss objective until the given threshold is reached for all tokens.
    """
    iteration = 0
    target_loss = max(0, 1.0 - threshold)
    while loss > target_loss:
        iteration += 1

        latents = latents.clone().detach().requires_grad_(True)

        loss = compute_loss(latents, ref_latents, mask, ref_mask)

        if loss != 0:
            latents = update_latent(latents, loss, step_size)

        print(f"\t Try {iteration}. loss: {loss}")

        if iteration >= max_refinement_steps:
            print(f"\t Exceeded max number of iterations ({max_refinement_steps})! ")
            break

    # Run one more time but don't compute gradients and update the latents.
    # We just need to compute the new loss - the grad update will occur below
    latents = latents.clone().detach().requires_grad_(True)
    loss = compute_loss(latents, ref_latents, mask, ref_mask)
    print(f"\t Finished with loss of: {loss}")
    return loss, latents

pretrained_model_name_or_path = "sd-legacy/stable-diffusion-inpainting"
# global_mapper_path = "/data1/lxy/project/generation/ELITE/elite_experiments/global_mapping/mapper_004000.pt"
global_mapper_path = "/data1/lxy/project/generation/ELITE_OURS/elite_experiments/mvtec_global_mapping_ref_random/mapper_020000.pt"
device = "cuda:0"
vae, unet, text_encoder, tokenizer, image_encoder, mapper, scheduler = pww_load_tools(
            device,
            LMSDiscreteScheduler,
            diffusion_model_path=pretrained_model_name_or_path,
            mapper_model_path=global_mapper_path,
        )

with torch.no_grad():

    scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )

    image_path = "/data1/lxy/project/generation/ELITE/datasets/mvtec/images/train/bottle/good/000.png"
    # image_path = "/data1/lxy/project/generation/ELITE/overture-creations-5sI6fQgYIuo.png"
    image_pil = Image.open(image_path)

    mask_path = "/data1/lxy/project/generation/ELITE/datasets/mvtec/masks/test/bottle/broken_small/006_mask.png"
    # mask_path = "/data1/lxy/project/generation/ELITE/overture-creations-5sI6fQgYIuo_mask.png"
    mask_pil = Image.open(mask_path)

    ref_image_path = "/data1/lxy/project/generation/ELITE/datasets/mvtec/images/test/bottle/broken_small/007.png"
    # ref_image_path = "/data1/lxy/project/generation/ELITE/test_datasets/1.jpg"
    ref_image_pil = Image.open(ref_image_path)

    ref_mask_path = "/data1/lxy/project/generation/ELITE/datasets/mvtec/masks/test/bottle/broken_small/007_mask.png"
    ref_mask_pil = Image.open(ref_mask_path)
    

    pixel_values_image = process_pil_image(image_pil)
    pixel_values_image = pixel_values_image.unsqueeze(0)
    pixel_values_mask = process_pil_mask(mask_pil)
    pixel_values_mask = pixel_values_mask.unsqueeze(0)
    pixel_values_masked_image = pixel_values_image * (pixel_values_mask < 0.5)

    pixel_values_ref_image_new = process_pil_image(ref_image_pil)
    pixel_values_ref_image_new = pixel_values_ref_image_new.unsqueeze(0)
    ref_image_np = np.array(ref_image_pil)

    pixel_values_ref_mask = process_pil_mask(ref_mask_pil)
    pixel_values_ref_mask = pixel_values_ref_mask.unsqueeze(0)

    ref_image_pil = Image.fromarray(ref_image_np.astype('uint8')).resize((224, 224), resample=Image.Resampling.BICUBIC)
    pixel_values_ref_image = get_tensor()(ref_image_pil)
    pixel_values_ref_image = pixel_values_ref_image.unsqueeze(0)





    seed = 47
    guidance_scale = 5

    pixel_values_image = pixel_values_image.to(device)
    pixel_values_mask = pixel_values_mask.to(device)
    pixel_values_masked_image = pixel_values_masked_image.to(device)
    pixel_values_ref_image_new = pixel_values_ref_image_new.to(device)
    pixel_values_ref_image = pixel_values_ref_image.to(device).half()
    pixel_values_ref_mask = pixel_values_ref_mask.to(device)
    uncond_input = tokenizer(
        [''] * pixel_values_image.shape[0],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_embeddings = text_encoder({'input_ids':uncond_input.input_ids.to(device)})[0]

    if seed is None:
        latents = torch.randn(
            (pixel_values_image.shape[0], 4, 64, 64)
        )
    else:
        generator = torch.manual_seed(seed)
        latents = torch.randn(
            (pixel_values_image.shape[0], 4, 64, 64), generator=generator,
        )

    latents = latents.to(pixel_values_ref_image)
    scheduler.set_timesteps(50)
    latents = latents * scheduler.init_noise_sigma

    placeholder_string = 'S'
    template="a photo of a S"
    text = template.format(placeholder_string)
    placeholder_index = 0
    words = text.strip().split(' ')
    for idx, word in enumerate(words):
        if word == placeholder_string:
            placeholder_index = idx + 1

    index = torch.tensor(placeholder_index)
    index = index.unsqueeze(0)
    index = index.to(device).long()

    input_ids = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids[0]
    input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)

    placeholder_idx = index

    masked_image_latents = vae.encode(pixel_values_masked_image.reshape(pixel_values_image.shape)).latent_dist.sample().detach()
    masked_image_latents = masked_image_latents * 0.18215


    pixel_values_mask = F.interpolate(pixel_values_mask, (64, 64), mode='bilinear')
    pixel_values_ref_mask = F.interpolate(pixel_values_ref_mask, (64, 64), mode='bilinear')

    ref_image_latents = vae.encode(F.interpolate(pixel_values_ref_image_new, (512, 512), mode='bilinear')).latent_dist.sample().detach()
    ref_image_latents = ref_image_latents * 0.18215
    ref_image = F.interpolate(pixel_values_ref_image, (224, 224), mode='bilinear')

    image_features = image_encoder(ref_image, output_hidden_states=True)
    image_embeddings = [image_features[0], image_features[2][4], image_features[2][8], image_features[2][12],
                        image_features[2][16]]
    image_embeddings = [emb.detach() for emb in image_embeddings]
    inj_embedding = mapper(image_embeddings)
    max_iter_to_alter = 50
    token_index = "0"
    if token_index != 'full':
        token_index = int(token_index)
        inj_embedding = inj_embedding[:, token_index:token_index + 1, :]

    encoder_hidden_states = text_encoder({'input_ids': input_ids,
                                            "inj_embedding": inj_embedding,
                                            "inj_index": placeholder_idx})[0]
    scale_range = np.linspace(1.0, 0.5, len(scheduler.timesteps))
    scale_factor = 100
    step_size = scale_factor * np.sqrt(scale_range)
    thresholds = {0: 0.05, 10: 0.5, 20: 0.8}
    for i, t in enumerate(tqdm(scheduler.timesteps)):
        with torch.enable_grad():
            latents = latents.clone().detach().requires_grad_(True)
            loss = compute_loss(latents, ref_image_latents, pixel_values_mask, pixel_values_ref_mask)
            if i in thresholds.keys() and loss > 1.0 - thresholds[i]:
                loss, latents = perform_iterative_refinement_step(
                    latents=latents,
                    ref_latents=ref_image_latents,
                    mask=pixel_values_mask,
                    ref_mask=pixel_values_ref_mask,
                    loss=loss,
                    threshold=thresholds[i],
                    step_size=step_size[i],
                )
            if i < max_iter_to_alter:
                if loss != 0:
                    latents = update_latent(latents=latents, loss=loss, step_size=step_size[i])
                    print(f'Iteration {i} | Loss: {loss:0.4f}')
        latent_model_input = scheduler.scale_model_input(latents, t)
        latent_model_input1 = latent_model_input
        latent_model_input1 = torch.cat([latent_model_input1, pixel_values_mask, masked_image_latents], dim=1)
        
        noise_pred_text = unet(
            latent_model_input1,
            t,
            encoder_hidden_states={
                "CONTEXT_TENSOR": encoder_hidden_states,
            }
        ).sample
        latent_model_input2 = latent_model_input
        latent_model_input2 = torch.cat([latent_model_input2, pixel_values_mask, masked_image_latents], dim=1)
        noise_pred_uncond = unet(
            latent_model_input2,
            t,
            encoder_hidden_states={
                "CONTEXT_TENSOR": uncond_embeddings,
            }
        ).sample
        noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
        )
        
        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
            




    _latents = 1 / 0.18215 * latents.clone()
    images = vae.decode(_latents).sample



    syn_images = [th2image(image) for image in images]

resize_mask_pil_np = np.array(mask_pil.resize((512, 512), resample=Image.Resampling.BICUBIC))
display_mask_np = np.stack([resize_mask_pil_np] * 3, axis=-1) if resize_mask_pil_np.ndim == 2 else resize_mask_pil_np
display_image_np = th2image(pixel_values_image[0])
display_ref_image_np = np.array(ref_image_pil.resize((512, 512), resample=Image.Resampling.BICUBIC))
resize_ref_mask_pil_np = np.array(ref_mask_pil.resize((512, 512), resample=Image.Resampling.BICUBIC))
display_ref_mask_np = np.stack([resize_ref_mask_pil_np] * 3, axis=-1) if resize_ref_mask_pil_np.ndim == 2 else resize_ref_mask_pil_np
display_syn_image_np = np.array(syn_images[0])

concat = np.concatenate((display_mask_np, display_image_np, display_ref_mask_np, display_ref_image_np, display_syn_image_np), axis=1)
save_dir = "/data1/lxy/project/generation/ELITE/my_test/images"

Image.fromarray(concat).save(os.path.join(save_dir, f'two_mask_newtt_90.png'))