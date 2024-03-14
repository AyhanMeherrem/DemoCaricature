import os

import numpy as np
import torch
import PIL.Image, PIL.ImageOps
from diffusers import T2IAdapter, EulerDiscreteScheduler
import gradio as gr

from handler import ExplicitROMEHandler
from pipeline import TextualStableDiffusionAdapterWithTauPipeline


MAX_SEED = np.iinfo(np.int32).max

IDENTITY_DICT = {
    "Barack Obama": {
        "path": "identities/Barack_Obama",
        "image": "./identity_images/Barack_Obama.jpg",
        "scale": 1.05,
    },
    "Whoopi Goldberg": {
        "path": "identities/Whoopi_Goldberg",
        "image": "./identity_images/Whoopi_Goldberg.jpg",
        "scale": .975,
    },
    "Mona Lisa": {
        "path": "identities/Mona_Lisa",
        "image": "./identity_images/Mona_Lisa.jpg",
        "scale": 0.95,
    },
    "Girl with a Pearl Earring": {
        "path": "identities/Girl_with_a_Pearl_Earring",
        "image": "./identity_images/Girl_with_a_Pearl_Earring.jpg",
        "scale": .975,
    },
    "Chow Yun-Fat": {
        "path": "identities/Chow_Yun-Fat",
        "image": "./identity_images/Chow_Yun-Fat.jpg",
        "scale": 1.125,
    },
    "George Clooney": {
        "path": "identities/George_Clooney",
        "image": "./identity_images/George_Clooney.jpg",
        "scale": 1.05,
    },
    "Mr. Bean": {
        "path": "identities/Mr_Bean",
        "image": "./identity_images/Mr_Bean.jpg",
        "scale": 1.15,
    },
    "Abraham Lincoln": {
        "path": "identities/Abraham_Lincoln",
        "image": "./identity_images/Abraham_Lincoln.jpg",
        "scale": .95,
    },
    "Anthony Hopkins": {
        "path": "identities/Anthony_Hopkins",
        "image": "./identity_images/Anthony_Hopkins.jpg",
        "scale": 1.1,
    },
    "Audrey Hepburn": {
        "path": "identities/Audrey_Hepburn",
        "image": "./identity_images/Audrey_Hepburn.jpg",
        "scale": .95,
    },
}

EXAMPLES = [
    [
        "Barack Obama",
        "./identity_images/Barack_Obama.jpg",
        "./assets/sketches/Barack_Obama.jpg",
        1.1,
        0.8,
        20,
        9,
        0.65,
        100,
    ],
]


def create_demo(handler: ExplicitROMEHandler) -> gr.Blocks:
    def run(
        sketch: PIL.Image.Image,
        identity_path: str,
        num_steps: int = 50,
        guidance_scale: float = 9,
        rome_scale: float = 1,
        features_adapter_tau: float = 0.65,
        features_adapter_weight: float = 0.7,
        seed: int = 100,
    ) -> PIL.Image.Image:
        if not check_inputs(sketch, identity_path):
            return None

        if handler.token is not None:
            handler.pipeline.unload_textual_inversion(tokens=handler.token)
        handler.load_explicit_rome(identity_path, token="<ID>")

        generator = torch.Generator("cuda").manual_seed(seed)
        print(rome_scale, features_adapter_weight, features_adapter_tau)

        return handler(
            prompt="a caricature of <ID>",
            image=sketch.convert("L"),
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            rome_scale=rome_scale,
            adapter_conditioning_scale=features_adapter_weight,
            adapter_conditioning_tau=features_adapter_tau,
            generator=generator,
        ).images[0]

    def process_example(
        identity: str,
        identity_path: str,
        sketch: PIL.Image.Image,
        rome_scale: float = 1,
        features_adapter_weight: float = 0.7,
        num_steps: int = 50,
        guidance_scale: float = 9,
        features_adapter_tau: float = 0.65,
        seed: int = 100,
    ) -> (PIL.Image.Image, PIL.Image.Image, str):
        sketch = PIL.ImageOps.invert(sketch)
        identity_path = IDENTITY_DICT[identity]["path"]
        result = run(
            sketch=sketch,
            identity_path=identity_path,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            rome_scale=rome_scale,
            features_adapter_tau=features_adapter_tau,
            features_adapter_weight=features_adapter_weight,
            seed=seed,
        )
        return result, None, identity_path

    def check_inputs(
        sketch: PIL.Image.Image,
        identity_path: str,
    ) -> bool:
        varify = True
        if sketch is None:
            gr.Warning("Draw a sketch!")
            varify = False
        if identity_path == "":
            gr.Warning("Select an Identity!")
            varify = False
        return varify

    def set_identity(
            identity
    ) -> (str, PIL.Image.Image, float):
        path = IDENTITY_DICT[identity]["path"]
        img = PIL.Image.open(IDENTITY_DICT[identity]["image"]).resize((512, 512))
        id_scale = IDENTITY_DICT[identity]["scale"]
        return path, img, id_scale

    def clean_result() -> PIL.Image.Image:
        return None

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                identity_path = gr.Textbox(visible=False)
                identity_img = gr.Image(
                    label="Identity Reference",
                    height=512,
                    width=512,
                )
                identity = gr.Dropdown(
                    choices=list(IDENTITY_DICT.keys()),
                    label="Select an Identity",
                )
            with gr.Column():
                with gr.Group():
                    sketch = gr.Image(
                        label="Sketch",
                        source="canvas",
                        tool="sketch",
                        type="pil",
                        image_mode="RGB",
                        invert_colors=True,
                        shape=(512, 512),
                        brush_radius=4,
                        height=512,
                        show_download_button=True,
                    )
                    with gr.Row():
                        rome_scale = gr.Slider(
                            label="ID scale",
                            minimum=0.,
                            maximum=2.,
                            step=0.025,
                            value=1,
                        )
                        adapter_conditioning_scale = gr.Slider(
                            label="Sketch scale",
                            minimum=0.,
                            maximum=1,
                            step=0.05,
                            value=0.7,
                        )
                    run_button = gr.Button("Run")
                with gr.Accordion("Advanced options", open=False):
                    num_steps = gr.Slider(
                        label="Number of steps",
                        minimum=1,
                        maximum=25,
                        step=1,
                        value=20,
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0.1,
                        maximum=15.0,
                        step=0.5,
                        value=9,
                    )
                    adapter_conditioning_tau = gr.Slider(
                        label="Sketch tau",
                        info="Fraction of timesteps for which sketch T2I-Adapter should be applied",
                        minimum=0.,
                        maximum=1,
                        step=0.05,
                        value=0.65,
                    )
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=100,
                    )
            with gr.Column():
                result = gr.Image(
                    label="Result",
                    type="pil",
                    height=512,
                    width=512,
                    show_share_button=True,
                    show_download_button=True,
                )

        hidden_sketch = gr.Image(
            label="Sketch",
            width=512,
            height=512,
            type="pil",
            visible=False,
        )
        gr.Examples(
            examples=EXAMPLES,
            run_on_click=True,
            inputs=[
                identity,
                identity_img,
                hidden_sketch,
                rome_scale,
                adapter_conditioning_scale,
                num_steps,
                guidance_scale,
                adapter_conditioning_tau,
                seed,
            ],
            outputs=[result, sketch, identity_path],
            fn=process_example,
        )

        inputs = [
            sketch,
            identity_path,
            num_steps,
            guidance_scale,
            rome_scale,
            adapter_conditioning_tau,
            adapter_conditioning_scale,
            seed,
        ]

        identity.select(
            fn=set_identity,
            inputs=[identity],
            outputs=[identity_path, identity_img, rome_scale],
        ).then(
            fn=clean_result,
            outputs=result,
        )

        run_button.click(
            fn=run,
            inputs=inputs,
            outputs=result,
            api_name=False,
        )

    return demo


if __name__ == "__main__":
    adapter = T2IAdapter.from_pretrained("TencentARC/t2iadapter_sketch_sd15v2", torch_dtype=torch.bfloat16)
    scheduler = EulerDiscreteScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    pipe = TextualStableDiffusionAdapterWithTauPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        adapter=adapter,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
        variant="fp16",
    ).to("cuda")
    pipe.safety_checker = None
    handler = ExplicitROMEHandler(pipe)

    demo = create_demo(handler)
    demo.queue(max_size=20).launch()
