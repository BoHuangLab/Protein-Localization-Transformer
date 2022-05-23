from math import log2, sqrt
import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange

from celle.vae import OpenAIDiscreteVAE, VQGanVAE
from celle.transformer import Transformer, DivideMax
import csv

from tqdm import tqdm

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class always:
    def __init__(self, val):
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return self.val


def is_empty(t):
    return t.nelement() == 0


def masked_mean(t, mask, dim=1):
    t = t.masked_fill(~mask[:, :, None], 0.0)
    return t.sum(dim=1) / mask.sum(dim=1)[..., None]


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


# sampling helpers


def log(t, eps=1e-20):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=0.9, dim=-1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(-1, ind, val)
    return probs


def typical(
    scores: torch.FloatTensor,
    mass: float = 0.9,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):

    # calculate entropy
    normalized = torch.nn.functional.log_softmax(scores, dim=-1)
    p = torch.exp(normalized)
    ent = -(normalized * p).nansum(-1, keepdim=True)

    # shift and sort
    shifted_scores = torch.abs((-normalized) - ent)
    sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
    sorted_logits = scores.gather(-1, sorted_indices)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative mass above the threshold
    last_ind = (cumulative_probs < mass).sum(dim=1)
    last_ind[last_ind < 0] = 0
    sorted_indices_to_remove = sorted_scores > sorted_scores.gather(
        1, last_ind.view(-1, 1)
    )
    if min_tokens_to_keep > 1:
        # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )

    scores = scores.masked_fill(indices_to_remove, filter_value)

    return scores


class ModelExtender(nn.Module):
    def __init__(self, vocab, out_features, fixed_embedding=False):
        super(ModelExtender, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.vocab = vocab
        if vocab == "unirep":
            from tape import UniRepModel

            self.model = UniRepModel.from_pretrained("babbler-1900")
            in_features = 1900
        elif vocab == "bert":
            from tape import ProteinBertModel

            self.model = ProteinBertModel.from_pretrained("bert-base")
            in_features = 768

        elif vocab == "esm1b":
            from esm import pretrained

            self.model, _ = pretrained.esm1b_t33_650M_UR50S()
            in_features = 33

        self.out_features = out_features
        self.scale_layer = nn.Linear(in_features, self.out_features)
        self.fixed_embedding = fixed_embedding
        if self.fixed_embedding:
            self.model = self.model.eval()

    def forward(self, x):

        if self.fixed_embedding:
            with torch.no_grad():
                if self.vocab == "esm1b":
                    self.model.eval()
                    x = self.model(x, repr_layers=[33])["representations"][33]
                else:
                    self.model.eval()
                    x = self.model(x)[0]
        else:
            if self.vocab == "esm1b":
                x = self.model(x, repr_layers=[33])["representations"][33]
            else:
                x = self.model(x)[0]

        if (
            (self.vocab == "unirep" and self.out_features != 1900)
            or (self.vocab == "bert" and self.out_features != 768)
            or (self.vocab == "esm1b" and self.out_features != 1280)
        ):
            x = self.scale_layer(x)

        return x


# discrete vae class


class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1),
        )

    def forward(self, x):
        return self.net(x) + x


class DiscreteVAE(nn.Module):
    def __init__(
        self,
        image_size=256,
        num_tokens=512,
        codebook_dim=512,
        num_layers=3,
        num_resnet_blocks=0,
        hidden_dim=64,
        channels=3,
        smooth_l1_loss=False,
        temperature=0.9,
        straight_through=False,
        kl_div_loss_weight=0.0,
        normalization=((0.5,) * 3, (0.5,) * 3),
    ):
        super().__init__()
        assert log2(image_size).is_integer(), "image size must be a power of 2"
        assert num_layers >= 1, "number of layers must be greater than or equal to 1"
        has_resblocks = num_resnet_blocks > 0

        self.image_size = image_size
        self.channels = channels
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.codebook = nn.Embedding(num_tokens, codebook_dim)

        hdim = hidden_dim

        enc_chans = [hidden_dim] * num_layers
        dec_chans = list(reversed(enc_chans))

        enc_chans = [channels, *enc_chans]

        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        enc_chans_io, dec_chans_io = map(
            lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans)
        )

        enc_layers = []
        dec_layers = []

        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            enc_layers.append(
                nn.Sequential(
                    nn.Conv2d(enc_in, enc_out, 4, stride=2, padding=1), nn.ReLU()
                )
            )
            dec_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(dec_in, dec_out, 4, stride=2, padding=1),
                    nn.ReLU(),
                )
            )

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))
            enc_layers.append(ResBlock(enc_chans[-1]))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv2d(codebook_dim, dec_chans[1], 1))

        enc_layers.append(nn.Conv2d(enc_chans[-1], num_tokens, 1))
        dec_layers.append(nn.Conv2d(dec_chans[-1], channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.kl_div_loss_weight = kl_div_loss_weight

        # take care of normalization within class
        self.normalization = normalization

        self._register_external_parameters()


    def norm(self, images):
        if not exists(self.normalization):
            return images

        means, stds = map(lambda t: torch.as_tensor(t).to(images), self.normalization)

        means, stds = map(lambda t: rearrange(t, "c -> () c () ()"), (means, stds))
        # means = torch.mean(means, axis=1)
        # stds = torch.std(stds, axis=1)
        images = images.clone()
        images.sub_(means).div_(stds)

        return images

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self(images, return_logits=True)
        codebook_indices = logits.argmax(dim=1).flatten(1)
        return codebook_indices

    def decode(self, img_seq):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))

        image_embeds = rearrange(image_embeds, "b (h w) d -> b d h w", h=h, w=w)
        images = self.decoder(image_embeds)
        return images

    def forward(
        self,
        img,
        return_loss=False,
        return_recons=False,
        return_logits=False,
        temp=None,
    ):
        device, num_tokens, image_size, kl_div_loss_weight = (
            img.device,
            self.num_tokens,
            self.image_size,
            self.kl_div_loss_weight,
        )
        assert (
            img.shape[-1] == image_size and img.shape[-2] == image_size
        ), f"input must have the correct image size {image_size}"

        img = self.norm(img)

        logits = self.encoder(img)

        if return_logits:
            return logits  # return logits for getting hard image indices for DALL-E training

        temp = default(temp, self.temperature)

        soft_one_hot = F.gumbel_softmax(
            logits, tau=temp, dim=1, hard=self.straight_through
        )

        sampled = einsum("b n h w, n d -> b d h w", soft_one_hot, self.codebook.weight)

        out = self.decoder(sampled)

        if not return_loss:
            return out

        # reconstruction loss

        recon_loss = self.loss_fn(img, out)

        # kl divergence

        logits = rearrange(logits, "b n h w -> b (h w) n")
        log_qy = F.log_softmax(logits, dim=-1)
        log_uniform = torch.log(torch.tensor([1.0 / num_tokens], device=device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, "batchmean", log_target=True)

        loss = recon_loss + (kl_div * kl_div_loss_weight)

        if not return_recons:
            return loss

        return loss, out




class OneHotEmbedding:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    # Defining __call__ method
    def __call__(self, sequence):
        return torch.nn.functional.one_hot(sequence, num_classes=self.num_classes)


class aaDescriptors:
    def __init__(self, path):
        with open(path, mode="r") as infile:
            reader = csv.reader(infile)
            self.embeddings = {
                int(rows[0]): torch.tensor(np.float_(rows[2:]))
                for rows in reader
                if rows[0] != "Index"
            }

    # Defining __call__ method
    def __call__(self, sequence):
        b, l = sequence.shape

        sequence = sequence.flatten()
        sequence = (
            torch.stack([self.embeddings[int(x.item())] for x in sequence], dim=0)
            .to(sequence.device)
            .float()
        )

        return sequence.view(b, l, 66)


class CELLE(nn.Module):
    def __init__(
        self,
        *,
        dim,
        vae,
        condition_vae=None,
        num_images=1,
        num_text_tokens=10000,
        text_seq_len=256,
        depth,
        heads=8,
        dim_head=64,
        reversible=False,
        attn_dropout=0.0,
        ff_dropout=0,
        sparse_attn=False,
        attn_types=None,
        loss_cond_weight=1,
        loss_img_weight=7,
        stable=False,
        sandwich_norm=False,
        shift_tokens=True,
        rotary_emb=True,
        text_embedding=None,
        fixed_embedding=False,
    ):
        super().__init__()
        assert isinstance(
            vae, (DiscreteVAE, VQGanVAE)
        ), "vae must be an instance of DiscreteVAE"

        self.text_embedding = text_embedding
        self.fixed_embedding = fixed_embedding

        if text_embedding is None:
            self.bos_token = num_text_tokens
            num_text_tokens += 1
            self.text_emb = nn.Embedding(num_text_tokens, dim)
            # num_text_tokens = num_text_tokens + text_seq_len

        elif text_embedding == "unirep":
            self.bos_token = 24
            self.text_emb = ModelExtender(text_embedding, dim, fixed_embedding)

        elif text_embedding == "bert":
            self.bos_token = 2
            self.text_emb = ModelExtender(text_embedding, dim, fixed_embedding)

        elif text_embedding == "esm1b":
            self.bos_token = 0
            self.text_emb = ModelExtender(text_embedding, dim, fixed_embedding)

        elif text_embedding == "onehot":
            self.bos_token = 24
            self.text_emb = OneHotEmbedding(num_classes=num_text_tokens)

        elif text_embedding == "aadescriptors":
            self.bos_token = 21
            self.text_emb = aaDescriptors(path="data/aaDescriptors.csv")

        self.text_pos_emb = (
            nn.Embedding(text_seq_len + 1, dim) if not rotary_emb else always(0)
        )  # +1 for <bos>

        self.num_text_tokens = num_text_tokens  # for offsetting logits index and calculating cross entropy loss
        self.text_seq_len = text_seq_len

        self.num_images = num_images
        image_size = vae.image_size
        num_image_tokens = vae.num_tokens
        image_fmap_size = vae.image_size // (2 ** vae.num_layers)
        image_seq_len = image_fmap_size ** 2

        self.image_emb = nn.Embedding(num_image_tokens, dim)

        self.image_pos_emb = (
            AxialPositionalEmbedding(
                dim, axial_shape=(image_fmap_size, image_fmap_size)
            )
            if not rotary_emb
            else always(0)
        )

        self.image_seq_len = image_seq_len

        self.num_image_tokens = num_image_tokens

        if exists(condition_vae):

            condition_size = condition_vae.image_size
            num_condition_tokens = condition_vae.num_tokens
            condition_fmap_size = condition_vae.image_size // (
                2 ** condition_vae.num_layers
            )
            condition_seq_len = condition_fmap_size ** 2

            self.condition_emb = nn.Embedding(num_condition_tokens, dim)

            self.condition_pos_emb = (
                AxialPositionalEmbedding(
                    dim, axial_shape=(condition_fmap_size, condition_fmap_size)
                )
                if not rotary_emb
                else always(0)
            )

        else:
            condition_fmap_size = 0
            condition_seq_len = 0
            num_condition_tokens = 0

        self.num_condition_tokens = num_condition_tokens
        self.condition_seq_len = condition_seq_len
        seq_len = text_seq_len + image_seq_len + condition_seq_len
        total_tokens = num_text_tokens + num_image_tokens + num_condition_tokens
        self.total_tokens = total_tokens
        self.total_seq_len = seq_len
        self.vae = vae
        self.condition_vae = condition_vae

        set_requires_grad(
            self.vae, self.vae_requires_grad
        )  # freeze VAE from being trained

        self.transformer = Transformer(
            dim=dim,
            causal=True,
            seq_len=seq_len,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            reversible=reversible,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            attn_types=attn_types,
            image_fmap_size=image_fmap_size + condition_fmap_size,
            num_images=num_images,
            sparse_attn=sparse_attn,
            stable=stable,
            sandwich_norm=sandwich_norm,
            shift_tokens=shift_tokens,
            rotary_emb=rotary_emb,
        )

        self.stable = stable

        if stable:
            self.norm_by_max = DivideMax(dim=-1)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.total_tokens),
        )

        seq_range = torch.arange(seq_len)
        logits_range = torch.arange(total_tokens)

        seq_range = rearrange(seq_range, "n -> () n ()")
        logits_range = rearrange(logits_range, "d -> () () d")

        if exists(condition_vae):
            logits_mask = (
                (seq_range >= text_seq_len) & (logits_range < num_text_tokens)
                | (seq_range >= text_seq_len)
                & (seq_range < text_seq_len + condition_seq_len)
                & (logits_range >= num_text_tokens + num_condition_tokens)
                | (seq_range >= text_seq_len + condition_seq_len)
                & (logits_range >= num_text_tokens)
                & (logits_range < num_text_tokens + num_condition_tokens)
                | ((seq_range < text_seq_len) & (logits_range >= num_text_tokens))
            )

        else:

            logits_mask = (
                (seq_range >= text_seq_len) & (logits_range < num_text_tokens)
            ) | ((seq_range < text_seq_len) & (logits_range >= num_text_tokens))

        self.register_buffer("logits_mask", logits_mask, persistent=False)
        self.loss_img_weight = loss_img_weight
        self.loss_cond_weight = loss_cond_weight

    @torch.no_grad()
    @eval_decorator
    def generate_images(
        self,
        text,
        *,
        clip=None,
        filter_thres=0.5,
        temperature=1.0,
        condition=None,
        img=None,
        num_init_img_tokens=None,
        return_logits=False,
        filter_method="top_k",
        progress=False,
    ):
        (
            vae,
            condition_vae,
            text_seq_len,
            image_seq_len,
            condition_seq_len,
            num_condition_tokens,
            num_text_tokens,
        ) = (
            self.vae,
            self.condition_vae,
            self.text_seq_len,
            self.image_seq_len,
            self.condition_seq_len,
            self.num_condition_tokens,
            self.num_text_tokens,
        )
        vae = vae.eval()
        if progress == True:
            progress = tqdm
        else:
            progress = lambda x: x

        total_len = text_seq_len + image_seq_len + condition_seq_len

        text = text[:, :text_seq_len]  # make sure text is within bounds
        out = text

        if exists(condition):
            with torch.no_grad():
                indices = condition_vae.get_codebook_indices(condition)
            indices = indices[:, :num_condition_tokens]
            out = torch.cat((out, indices), dim=-1)

        if exists(img):
            with torch.no_grad():
                indices = vae.get_codebook_indices(img)
            num_img_tokens = default(
                num_init_img_tokens, int(0.4375 * image_seq_len)
            )  # OpenAI used 14 * 32 initial tokens to prime
            # assert num_img_tokens < image_seq_len, 'number of initial image tokens for priming must be less than the total image token sequence length'
            indices = indices[:, :num_img_tokens]
            out = torch.cat((out, indices), dim=-1)

        for cur_len in progress(range(out.shape[1], total_len)):

            is_image = cur_len >= text_seq_len
            if is_image:
                is_not_condition = cur_len >= (text_seq_len + condition_seq_len)

            (text, condition, image) = (
                out[:, :text_seq_len],
                out[:, text_seq_len : text_seq_len + condition_seq_len],
                out[:, text_seq_len + condition_seq_len :],
            )

            logits = self(text=text, condition=condition, image=image)
            full_logits = logits

            logits = logits[:, -1, :]

            if filter_method == "top_k":
                filtered_logits = top_k(logits, thres=filter_thres)
            elif filter_method == "typical":
                filtered_logits = typical(logits, min_tokens_to_keep=2)
            sample = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            sample -= (
                (num_text_tokens) if is_image else 0
            )  # offset sampled token if it is an image token, since logit space is composed of text and then image tokens
            sample -= (
                (num_condition_tokens) if is_not_condition else 0
            )  # offset sampled token if it is a condition token, since logit space is composed of text and then image tokens

            out = torch.cat((out, sample[:, None]), dim=-1)

        text_seq = out[:, :text_seq_len]
        condition_seq = out[:, text_seq_len : text_seq_len + condition_seq_len]
        img_seq = out[:, text_seq_len + condition_seq_len :]

        images = vae.decode(img_seq)
        if return_logits:
            return images, full_logits
        if exists(clip):
            scores = clip(text_seq, images, return_loss=False)
            return images, scores

        return images

    def forward(
        self, text, condition=None, image=None, return_loss=False, return_encoding=False
    ):

        assert (
            text.shape[-1] == self.text_seq_len
        ), f"the length {text.shape[-1]} of the text tokens you passed in does not have the correct length ({self.text_seq_len})"
        device, total_seq_len = text.device, self.total_seq_len

        self.image = image
        self.condition = condition

        # add <bos>

        text = F.pad(text, (1, 0), value=self.bos_token)

        tokens = self.text_emb(text)

        tokens += self.text_pos_emb(torch.arange(text.shape[1], device=device))

        seq_len = tokens.shape[1]

        if exists(condition) and not is_empty(condition) and exists(self.condition_vae):

            is_raw_image = len(condition.shape) == 4
            if is_raw_image:
                with torch.no_grad():
                    condition = self.condition_vae.get_codebook_indices(condition)

            condition_len = condition.shape[1]
            condition_emb = self.condition_emb(condition)
            condition_emb += self.condition_pos_emb(condition_emb)
            tokens = torch.cat((tokens, condition_emb), dim=1)
            seq_len += condition_len

        if exists(image) and not is_empty(image) and exists(self.vae):

            is_raw_image = len(image.shape) == 4

            if is_raw_image:
                image_size = self.vae.image_size
                # assert tuple(image.shape[1:]) == (self.vae.channels, image_size, image_size), f'invalid image of dimensions {image.shape} passed in during training'
                if self.vae_requires_grad:
                    image = self.vae.get_codebook_indices(image)

            image_emb = self.image_emb(image)
            image_emb += self.image_pos_emb(image_emb)
            image_len = image.shape[1]
            tokens = torch.cat((tokens, image_emb), dim=1)
            seq_len += image_len

        # when training, if the length exceeds the total text + image length
        # remove the last token, since it needs not to be trained

        if tokens.shape[1] > total_seq_len:
            seq_len -= 1
            tokens = tokens[:, :-1]

        if self.stable:
            alpha = 0.1
            tokens = tokens * alpha + tokens.detach() * (1 - alpha)

        out = self.transformer(tokens)

        if self.stable:
            out = self.norm_by_max(out)

        if return_encoding:
            return out

        logits = self.to_logits(out)

        # mask logits to make sure text predicts text (except last token), and image predicts image
        logits_mask = self.logits_mask[:, :seq_len]

        max_neg_value = -torch.finfo(logits.dtype).max
        logits.masked_fill_(logits_mask, max_neg_value)

        if not return_loss:
            return logits

        labels = text[:, 1:]

        if exists(condition) and exists(self.condition_vae):
            offsetted_condition = condition + self.num_text_tokens
            labels = torch.cat((labels, offsetted_condition), dim=1)

        assert exists(image), "when training, image must be supplied"

        offsetted_image = image + self.num_text_tokens + self.num_condition_tokens

        labels = torch.cat((labels, offsetted_image), dim=1)

        logits_re = rearrange(logits, "b n c -> b c n")

        if self.text_seq_len > 1:

            loss_text = F.cross_entropy(
                logits_re[:, :, : self.text_seq_len], labels[:, : self.text_seq_len]
            )

        else:
            loss_text = 0

        loss_cond = F.cross_entropy(
            logits_re[
                :,
                :,
                self.text_seq_len : self.text_seq_len + self.condition_seq_len,
            ],
            labels[:, self.text_seq_len : self.text_seq_len + self.condition_seq_len],
        )

        loss_img = F.cross_entropy(
            logits_re[:, :, self.text_seq_len + self.condition_seq_len :],
            labels[:, self.text_seq_len + self.condition_seq_len :],
        )

        loss_vae_cond = self.vae(self.image, 1)
        loss_vae_img = self.vae(self.condition, 1)

        loss_dict = {
            "loss_text": loss_text,
            "loss_cond": loss_cond,
            "loss_img": loss_img,
            "loss_vae_cond": loss_vae_cond,
            "loss_vae_img": loss_vae_img,
        }

        loss = (
            loss_text
            + self.loss_cond_weight * (loss_cond + loss_vae_cond * 10)
            + self.loss_img_weight * (loss_img + loss_vae_img * 10)
        ) / (self.loss_img_weight + self.loss_cond_weight + 1)

        return loss, loss_dict, logits
