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

pretrained_model_name_or_path = "sd-legacy/stable-diffusion-inpainting"
global_mapper_path = "elite_experiments/mvtec_global_mapping/mapper.pt"
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

    image_path = "datasets/mvtec_defect_collection/images/bottle_broken_large_000.png"
    image_pil = Image.open(image_path)

    mask_path = "datasets/mvtec_defect_collection/masks/bottle_broken_large_000_mask.png"
    mask_pil = Image.open(mask_path)

    ref_image_path = "datasets/mvtec_defect_collection/images/bottle_broken_large_000.png"
    ref_image_pil = Image.open(ref_image_path)


    pixel_values_image = process_pil_image(image_pil)
    pixel_values_image = pixel_values_image.unsqueeze(0)
    pixel_values_mask = process_pil_mask(mask_pil)
    pixel_values_mask = pixel_values_mask.unsqueeze(0)
    pixel_values_masked_image = pixel_values_image * (pixel_values_mask < 0.5)


    ref_image_np = np.array(ref_image_pil)



    ref_image_pil = Image.fromarray(ref_image_np.astype('uint8')).resize((224, 224), resample=Image.Resampling.BICUBIC)
    pixel_values_ref_image = get_tensor()(ref_image_pil)
    pixel_values_ref_image = pixel_values_ref_image.unsqueeze(0)





    seed = 47
    guidance_scale = 5

    pixel_values_image = pixel_values_image.to(device)
    pixel_values_mask = pixel_values_mask.to(device)
    pixel_values_masked_image = pixel_values_masked_image.to(device)
    pixel_values_ref_image = pixel_values_ref_image.to(device).half()

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
    scheduler.set_timesteps(100)
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


    image = F.interpolate(pixel_values_ref_image, (224, 224), mode='bilinear')

    image_features = image_encoder(image, output_hidden_states=True)
    image_embeddings = [image_features[0], image_features[2][4], image_features[2][8], image_features[2][12],
                        image_features[2][16]]
    image_embeddings = [emb.detach() for emb in image_embeddings]
    inj_embedding = mapper(image_embeddings)

    token_index = "0"
    if token_index != 'full':
        token_index = int(token_index)
        inj_embedding = inj_embedding[:, token_index:token_index + 1, :]

    encoder_hidden_states = text_encoder({'input_ids': input_ids,
                                            "inj_embedding": inj_embedding,
                                            "inj_index": placeholder_idx})[0]

    for t in tqdm(scheduler.timesteps):
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
display_syn_image_np = np.array(syn_images[0])

concat = np.concatenate((display_mask_np, display_image_np, display_ref_image_np, display_syn_image_np), axis=1)
save_dir = "images"
os.makedirs(save_dir, exist_ok=True)
Image.fromarray(concat).save(os.path.join(save_dir, f'generated_image_without_gradient.png'))