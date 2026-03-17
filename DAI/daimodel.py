import os
import time

from tqdm import tqdm
import torch
import torch.nn as nn
from DAI.controlnetvae import ControlNetVAEModel
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, AutoTokenizer
from DAI.decoder import CustomAutoencoderKL
from DAI.pipeline_all import DAIPipeline
from DAI.unet import UNet2DConditionModel
from PIL import Image
import numpy as np
from diffusers.utils import make_image_grid, load_image
import random


def extract_into_tensor(arr, timesteps, broadcast_shape):
    timesteps = timesteps.to(arr.device, dtype=torch.long)
    res = arr.gather(dim=0, index=timesteps)
    while len(res.shape) < len(broadcast_shape):
        res = res.unsqueeze(-1)
    return res.expand(broadcast_shape)


class DAIModel(torch.nn.Module):
    def __init__(self, args, mode="train"):
        super().__init__()
        assert mode in ["train", "inference"]

        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.pretrained_dai, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.args.pretrained_dai, subfolder="text_encoder"
        )
        self.scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_dai, subfolder="scheduler"
        )
        self.scheduler.set_timesteps(4)
        self.scheduler.prediction_type = 'sample'
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            self.args.pretrained_dai, subfolder="vae"
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.args.pretrained_dai, subfolder="unet"
        )
        self.vae_2 = CustomAutoencoderKL.from_pretrained(
            args.pretrained_dai, subfolder="cross_vae"
        )
        if mode == "inference":
            if self.args.controlnet is not None:
                print("===> Load controlnet {} <===".format(self.args.controlnet))
                checkpoint = torch.load(self.args.controlnet, map_location="cpu")
                self.unet.load_state_dict(checkpoint['model'])
                self.controlnet = ControlNetVAEModel.from_unet(self.unet)
                self.controlnet.load_state_dict(checkpoint['controlnet'])
            if self.args.cross_vae is not None:
                print("===> Load cross vae {} <===".format(self.args.cross_vae))
                self.vae_2 = CustomAutoencoderKL.from_pretrained(self.args.cross_vae)
        self.unet.to("cuda", dtype=self.weight_dtype)
        self.vae.to("cuda", dtype=self.weight_dtype)
        self.text_encoder.to("cuda", dtype=self.weight_dtype)
        self.vae_2.to("cuda", dtype=self.weight_dtype)
        self.controlnet.to("cuda", dtype=self.weight_dtype)

        if mode == "inference":
            self.pipeline = DAIPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=self.unet,
                controlnet=(
                    self.controlnet if self.args.controlnet is not None else None
                ),
                safety_checker=None,
                scheduler=None,
                feature_extractor=None,
                t_start=0,
            ).to(self.device)
        else:
            self.init_loss()

    def load_controlnet(self, model_path):
        self.controlnet = ControlNetVAEModel.from_pretrained(
            model_path, torch_dtype=torch.float32
        ).to(self.device)

    def inference(self, input_dir, output_dir, concat_dir, processing_resolution=1024):
        all_inputs = os.listdir(input_dir)
        for input_name in tqdm(all_inputs):
            inp_path = os.path.join(input_dir, input_name)
            input_image = load_image(inp_path)
            orig_size = input_image.size

            with torch.no_grad():
                result_image = self.pipeline(
                    image=torch.tensor(np.array(input_image))
                    .permute(2, 0, 1)
                    .float()
                    .div(255)
                    .unsqueeze(0)
                    .to(self.device),
                    prompt="remove shadow pattern, high quality, detailed",
                    vae_2=self.vae_2,
                    processing_resolution=processing_resolution,
                ).prediction[0]

            result_image = (result_image.clip(-1, 1) + 1) / 2
            result_image = result_image * 255
            result_image = result_image.astype(np.uint8)
            result_image = Image.fromarray(result_image)
            concat_image = make_image_grid([input_image, result_image], rows=1, cols=2)

            # Save the concatenated image
            concat_out_path = os.path.join(concat_dir, f"{input_name}")
            result_out_path = os.path.join(output_dir, f"{input_name}")
            concat_image.save(concat_out_path)
            result_image.save(result_out_path)
            print("save concat_out at {}".format(concat_out_path))
            print("save result_out at {}".format(result_out_path))

    def ddimsample(self, noisy_latents, text_embeddings):
        for t in self.scheduler.timesteps:
            print("time_step: ", t)
            output = self.unet(noisy_latents,
                                   t,
                                   encoder_hidden_states=text_embeddings.repeat(noisy_latents.shape[0], 1, 1),
                                   return_dict=False,)[0]
            noisy_latents = self.scheduler.step(model_output=output, timestep=t, sample=noisy_latents).prev_sample
        return noisy_latents

