import torch
from diffusers import T2IAdapter, EulerDiscreteScheduler # T2I: Network that extract features from sketch, EulerSchedular: Differential equation solver that calculates how much noise will be reduced on iterative denoising step. We Could use DDIMSchedular or else
from PIL import Image, ImageOps

from handler import ExplicitROMEHandler # Local algorithm that writes persons identity feature vectors directly inside difussions cross attention matrixes
from pipeline import TextualStableDiffusionAdapterWithTauPipeline


model_name = "runwayml/stable-diffusion-v1-5" 
adapter_name = "TencentARC/t2iadapter_sketch_sd15v2"

adapter = T2IAdapter.from_pretrained(adapter_name, torch_dtype=torch.bfloat16)
scheduler = EulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")
pipe = TextualStableDiffusionAdapterWithTauPipeline.from_pretrained(
    model_name,
    adapter=adapter,
    scheduler=scheduler,
    dtype=torch.bfloat16,
    variant="fp16",
).to("cuda")
pipe.safety_checker = None
handler = ExplicitROMEHandler(pipe)
handler.load_explicit_rome("identities/Barack_Obama", token="<ID>")

sketch = Image.open("assets/sketches/Barack_Obama.jpg").convert("L")  #grayscale
sketch = ImageOps.invert(sketch)  #Moddel trained on black background 

#generator = torch.Generator("cuda").manual_seed(100)
#sample = handler(
#    prompt="a caricature of <ID>",
#    image=sketch,
#    num_inference_steps=20,
#    guidance_scale=9,o
#    rome_scale=1.1,
#    adapter_conditioning_scale=0.8,
#    adapter_conditioning_tau=0.65,
#    generator=generator,
#).images[0]
#sample.save("Barack_Obama_caricature.jpg")

# To understand shape-ident tradeoff we need to change them only while preserving others 

rome_scales = [0.8, 1.0, 1.2]
shape_scales = [0.5, 0.8, 1.0]

for r_scale in rome_scales:
    for s_scale in shape_scales:
        sample = handler(
            prompt="a caricature of <ID>",
            image=sketch,
            num_inference_steps=20,
            guidance_scale=9,
            rome_scale=r_scale,
            adapter_conditioning_scale=s_scale,
            adapter_conditioning_tau=0.65,  # Tau: It determines when to remove skecth from iteration, 20x0.65=13 so 1-13 will be with skecth and 13-20 without
            generator=generator,
        ).images[0]
        
        filename = f"output_rome{r_scale}_shape{s_scale}.png"
        sample.save(filename)
