import torch
import torch.nn as nn
import kornia
import open_clip
from torch.utils.checkpoint import checkpoint
from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel
from lvdm.common import autocast
from utils.utils import count_params

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()
    def encode(self, *args, **kwargs):
        raise NotImplementedError

class IdentityEncoder(AbstractEncoder):
    def encode(self, x):
        return x

class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class', ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate
    def forward(self, batch, key=None, disable_dropout=False):
        if key is None: key = self.key
        c = batch[key][:, None]
        if self.ucg_rate > 0. and not disable_dropout:
            mask = 1. - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1 - mask) * torch.ones_like(c) * (self.n_classes - 1)
            c = c.long()
        c = self.embedding(c)
        return c
    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc

def disabled_train(self, mode=True):
    return self

class FrozenT5Embedder(AbstractEncoder):
    def __init__(self, version="google/t5-v1_1-large", device="cuda", max_length=77, freeze=True):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze: self.freeze()
    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters(): param.requires_grad = False
    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        return z
    def encode(self, text):
        return self(text)

class FrozenCLIPEmbedder(AbstractEncoder):
    LAYERS = ["last", "pooled", "hidden"]
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77, freeze=True, layer="last", layer_idx=None):
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze: self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12
    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters(): param.requires_grad = False
    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer == "hidden")
        if self.layer == "last": z = outputs.last_hidden_state
        elif self.layer == "pooled": z = outputs.pooler_output[:, None, :]
        else: z = outputs.hidden_states[self.layer_idx]
        return z
    def encode(self, text):
        return self(text)

class ClipImageEmbedder(nn.Module):
    def __init__(self, model, jit=False, device='cuda' if torch.cuda.is_available() else 'cpu', antialias=True, ucg_rate=0.):
        super().__init__()
        from clip import load as load_clip
        self.model, _ = load_clip(name=model, device=device, jit=jit)
        self.antialias = antialias
        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)
        self.ucg_rate = ucg_rate
    def preprocess(self, x):
        x = kornia.geometry.resize(x, (224, 224), interpolation='bicubic', align_corners=True, antialias=self.antialias)
        x = (x + 1.) / 2.
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x
    def forward(self, x, no_dropout=False):
        out = self.model.encode_image(self.preprocess(x))
        out = out.to(x.dtype)
        if self.ucg_rate > 0. and not no_dropout:
            out = torch.bernoulli((1. - self.ucg_rate) * torch.ones(out.shape[0], device=out.device))[:, None] * out
        return out

class FrozenOpenCLIPEmbedder(AbstractEncoder):
    LAYERS = ["last", "penultimate"]
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77, freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model
        self.device = device
        self.max_length = max_length
        if freeze: self.freeze()
        self.layer = layer
        if self.layer == "last": self.layer_idx = 0
        elif self.layer == "penultimate": self.layer_idx = 1
        else: raise NotImplementedError()
    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters(): param.requires_grad = False
    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z
    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.model.ln_final(x)
        return x
    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx: break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting(): x = checkpoint(r, x, attn_mask)
            else: x = r(x, attn_mask=attn_mask)
        return x
    def encode(self, text):
        return self(text)

class FrozenOpenCLIPImageEmbedder(AbstractEncoder):
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77, freeze=True, layer="pooled", antialias=True, ucg_rate=0.):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.transformer
        self.model = model
        self.device = device
        self.max_length = max_length
        if freeze: self.freeze()
        self.layer = layer
        if self.layer == "penultimate": raise NotImplementedError()
        self.antialias = antialias
        # THE FIX IS HERE
        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1), persistent=False)
        self.ucg_rate = ucg_rate
    def preprocess(self, x):
        x = kornia.geometry.resize(x, (224, 224), interpolation='bicubic', align_corners=True, antialias=self.antialias)
        x = (x + 1.) / 2.
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x
    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters(): param.requires_grad = False
    @autocast
    def forward(self, image, no_dropout=False):
        z = self.encode_with_vision_transformer(image)
        if self.ucg_rate > 0. and not no_dropout:
            z = torch.bernoulli((1. - self.ucg_rate) * torch.ones(z.shape[0], device=z.device))[:, None] * z
        return z
    def encode_with_vision_transformer(self, img):
        img = self.preprocess(img)
        x = self.model.visual(img)
        return x
    def encode(self, text):
        return self(text)

class FrozenOpenCLIPImageEmbedderV2(AbstractEncoder):
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", freeze=True, layer="pooled", antialias=True):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.transformer
        self.model = model
        self.device = device
        if freeze: self.freeze()
        self.layer = layer
        if self.layer == "penultimate": raise NotImplementedError()
        self.antialias = antialias
        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)
    def preprocess(self, x):
        x = kornia.geometry.resize(x, (224, 224), interpolation='bicubic', align_corners=True, antialias=self.antialias)
        x = (x + 1.) / 2.
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x
    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters(): param.requires_grad = False
    def forward(self, image, no_dropout=False): 
        z = self.encode_with_vision_transformer(image)
        return z
    def encode_with_vision_transformer(self, x):
        x = self.preprocess(x)
        if self.model.visual.input_patchnorm:
            x = x.reshape(x.shape[0], x.shape[1], self.model.visual.grid_size[0], self.model.visual.patch_size[0], self.model.visual.grid_size[1], self.model.visual.patch_size[1])
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.model.visual.grid_size[0] * self.model.visual.grid_size[1], -1)
            x = self.model.visual.patchnorm_pre_ln(x)
            x = self.model.visual.conv1(x)
        else:
            x = self.model.visual.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
        x = torch.cat([self.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.model.visual.positional_embedding.to(x.dtype)
        x = self.model.visual.patch_dropout(x)
        x = self.model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.model.visual.transformer(x)
        x = x.permute(1, 0, 2)
        return x

class FrozenCLIPT5Encoder(AbstractEncoder):
    def __init__(self, clip_version="openai/clip-vit-large-patch14", t5_version="google/t5-v1_1-xl", device="cuda", clip_max_length=77, t5_max_length=77):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(clip_version, device, max_length=clip_max_length)
        self.t5_encoder = FrozenT5Embedder(t5_version, device, max_length=t5_max_length)
        print(f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder) * 1.e-6:.2f} M parameters, "
              f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder) * 1.e-6:.2f} M params.")
    def encode(self, text):
        return self(text)
    def forward(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]