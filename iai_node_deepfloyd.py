from typing import Literal, Optional
from pydantic import BaseModel, Field
from .baseinvocation import BaseInvocation, InvocationContext, BaseInvocationOutput
from .image import ImageOutput, build_image_output

from diffusers import DiffusionPipeline
import torch

from ..models.image import ImageType
from .latent import LatentsOutput, LatentsField, random_seed

from ...backend.util.util import image_to_dataURL
from invokeai.app.api.models.images import ProgressImage
from diffusers.utils import pt_to_pil

"""
TODO:
Features:
- i2i and inpainting
- Select box for model - Literal[tuple("item1", "item2")] = Field(default="item1", description="description")
- Add comment/readme with directions for license and pip requirements
- height
- width

Code Re-Use:
- Adapt Compel node to handle T5
- Adapt Model Manager to handle models
- Figure out if stage 3 can be implemented generically

Completeness:
- Schema_extra stuff
- Full Metadata?
- Safety modules?
- How to do dtype and variant correctly

- Can each DF take in noise like the t2l node, or does that need be a generator?
"""

# Partially taken from latent.py
def dispatch_progress(
    self: any, context: InvocationContext, step: int, latents: torch.FloatTensor
) -> None:
    graph_execution_state = context.services.graph_execution_manager.get(context.graph_execution_state_id)
    source_node_id = graph_execution_state.prepared_source_mapping[self.id]

    image = pt_to_pil(latents)[0]

    (width, height) = image.size
    width *= 8
    height *= 8

    #TODO: This conversion is wrong.
    if image.mode in ["RGBA", "P"]:
        image = image.convert("RGB")

    dataURL = image_to_dataURL(image, image_format="JPEG")

    context.services.events.emit_generator_progress(
        graph_execution_state_id=context.graph_execution_state_id,
        node=self.dict(),
        source_node_id=source_node_id,
        progress_image=ProgressImage(width=width, height=height, dataURL=dataURL),
        step=step,
        total_steps=self.steps,
    )

class PromptEmbedsOutput(BaseInvocationOutput):
    # fmt: off
    type: Literal["latents_pair"] = "latents_pair"
    prompt_embeds: LatentsField = Field(default=None, description="Latents #1")
    negative_embeds: LatentsField = Field(default=None, description="Latents #2")
    # fmt: on

class PromptEmbedsInvocation(BaseInvocation):
    type: Literal["prompt_embeds"] = "prompt_embeds"
    prompt: str = Field(default=None, description="The input prompt")
    negative_prompt: str = Field(default=None, description="The negative prompt")
    stage_1_model: str = Field(default="DeepFloyd/IF-I-XL-v1.0", description="The stage 1 model")
    enable_cpu_offload: bool = Field(default=True, description="Enable CPU offload")

    def invoke(self, context: InvocationContext) -> PromptEmbedsOutput:
        dtype = torch.float16
        variant = "fp16"

        stage_1 = DiffusionPipeline.from_pretrained(self.stage_1_model, variant=variant, torch_dtype=dtype, unet=None)
        
        if self.enable_cpu_offload:
            stage_1.enable_model_cpu_offload()
        else:
            stage_1.to("cuda")

        prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt=self.prompt, negative_prompt=self.negative_prompt)

        name1 = f'{context.graph_execution_state_id}__{self.id}_prompt'
        context.services.latents.set(name1, prompt_embeds)

        name2 = f'{context.graph_execution_state_id}__{self.id}_negative_prompt'
        context.services.latents.set(name2, negative_embeds)

        return PromptEmbedsOutput(prompt_embeds=LatentsField(latents_name=name1), negative_embeds=LatentsField(latents_name=name2))

class DeepFloydStage1Invocation(BaseInvocation):
    #fmt: off
    type: Literal["deep_floyd_stage_1"] = "deep_floyd_stage_1"
    prompt_embeds: LatentsField = Field(default=None, description="The input prompt")
    negative_embeds: LatentsField = Field(default=None, description="The input negative prompt")
    stage_1_model: str = Field(default="DeepFloyd/IF-I-XL-v1.0", description="The stage1 model")
    enable_cpu_offload: bool = Field(default=False, description="Enable CPU offload")
    seed: int = Field(default=0, description="The seed to use for generation")
    steps:       int = Field(default=50, gt=0, description="The number of steps to use to generate the image")
    cfg_scale: float = Field(default=7.5, gt=0, description="The Classifier-Free Guidance, higher values may result in a result closer to the prompt", )    
    #fmt: on

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        dtype = torch.float16
        variant = "fp16"
        prompt_embeds = context.services.latents.get(self.prompt_embeds.latents_name)
        negative_embeds = context.services.latents.get(self.negative_embeds.latents_name)
        seed = self.seed if self.seed != -1 else random_seed()

        stage_1 = DiffusionPipeline.from_pretrained(self.stage_1_model, variant=variant, torch_dtype=dtype, text_encoder=None)
        
        if self.enable_cpu_offload:
            stage_1.enable_model_cpu_offload()
        else:
            stage_1.to("cuda")

        def step_callback(step: int, timestep: int, latents: torch.FloatTensor):
            dispatch_progress(self=self, context=context, step=step, latents=latents)

        generator = torch.manual_seed(seed)
        image = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt", callback=step_callback, num_inference_steps=self.steps, guidance_scale=self.cfg_scale).images

        image_type = ImageType.RESULT
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )
        metadata = context.services.metadata.build_metadata(
            session_id=context.graph_execution_state_id, node=self
        )
        context.services.images.save(image_type, image_name, pt_to_pil(image)[0], metadata)

        name = f'{context.graph_execution_state_id}__{self.id}'
        context.services.latents.set(name, image)
        return LatentsOutput(latents=LatentsField(latents_name=name))


class DeepFloydStage2Invocation(BaseInvocation):
    #fmt: off
    type: Literal["deep_floyd_stage_2"] = "deep_floyd_stage_2"
    prompt_embeds: LatentsField = Field(default=None, description="The input prompt")
    negative_embeds: LatentsField = Field(default=None, description="The input negative prompt")
    latents: LatentsField = Field(default=None, description="The latents to generate an image from")
    stage_2_model: str = Field(default="DeepFloyd/IF-II-L-v1.0", description="The stage2 model")
    enable_cpu_offload: bool = Field(default=False, description="Enable CPU offload")
    seed: int = Field(default=0, description="The seed to use for generation")
    steps:       int = Field(default=50, gt=0, description="The number of steps to use to generate the image")
    cfg_scale: float = Field(default=7.5, gt=0, description="The Classifier-Free Guidance, higher values may result in a result closer to the prompt", )    
    #fmt: on

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        dtype = torch.float16
        variant = "fp16"
        latents = context.services.latents.get(self.latents.latents_name)
        prompt_embeds = context.services.latents.get(self.prompt_embeds.latents_name)
        negative_embeds = context.services.latents.get(self.negative_embeds.latents_name)
        seed = self.seed if self.seed != -1 else random_seed()

        stage_2 = DiffusionPipeline.from_pretrained(self.stage_2_model, text_encoder=None, variant=variant, torch_dtype=dtype, num_inference_steps=self.steps, guidance_scale=self.cfg_scale)

        if self.enable_cpu_offload:
            stage_2.enable_model_cpu_offload()
        else:   
            stage_2.to("cuda")

        generator = torch.manual_seed(seed)

        def step_callback(step: int, timestep: int, latents: torch.FloatTensor):
            dispatch_progress(self=self, context=context, step=step, latents=latents)

        image = stage_2(
            image=latents, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt", callback=step_callback
        ).images

        image_type = ImageType.RESULT
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )
        metadata = context.services.metadata.build_metadata(
            session_id=context.graph_execution_state_id, node=self
        )
        context.services.images.save(image_type, image_name, pt_to_pil(image)[0], metadata)

        name = f'{context.graph_execution_state_id}__{self.id}'
        context.services.latents.set(name, image)
        return LatentsOutput(latents=LatentsField(latents_name=name))

class DeepFloydStage3Invocation(BaseInvocation):
    #fmt: off
    type: Literal["deep_floyd_stage_3"] = "deep_floyd_stage_3"
    latents: LatentsField = Field(default=None, description="The latents to generate an image from")
    prompt: str = Field(default=None, description="The input prompt")
    negative_prompt: str = Field(default=None, description="The negative prompt")
    stage_3_model: str = Field(default="stabilityai/stable-diffusion-x4-upscaler", description="The stage 3 model")
    noise_level: int = Field(default=100, description="The noise level")
    enable_cpu_offload: bool = Field(default=False, description="Enable CPU offload")
    seed: int = Field(default=0, description="The seed to use for generation")
    steps:       int = Field(default=50, gt=0, description="The number of steps to use to generate the image")
    cfg_scale: float = Field(default=7.5, gt=0, description="The Classifier-Free Guidance, higher values may result in a result closer to the prompt", )    
    #fmt: on

    def invoke(self, context: InvocationContext) -> ImageOutput:
        dtype = torch.float16
        latents = context.services.latents.get(self.latents.latents_name)

        stage_3 = DiffusionPipeline.from_pretrained(self.stage_3_model, torch_dtype=dtype)

        if self.enable_cpu_offload:
            stage_3.enable_model_cpu_offload()
        else:
            stage_3.to("cuda")

        generator = torch.manual_seed(self.seed)

        def step_callback(step: int, timestep: int, latents: torch.FloatTensor):
            dispatch_progress(self=self, context=context, step=step, latents=latents)

        image = stage_3(prompt=self.prompt, negative_prompt=self.negative_prompt, image=latents, generator=generator, noise_level=self.noise_level, callback=step_callback, num_inference_steps=self.steps, guidance_scale=self.cfg_scale).images


        image_type = ImageType.RESULT
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )
        metadata = context.services.metadata.build_metadata(
            session_id=context.graph_execution_state_id, node=self
        )
        context.services.images.save(image_type, image_name, image[0], metadata)

        #TODO: Should I be doing this elsewhere too?
        torch.cuda.empty_cache()

        return build_image_output(
            image_type=image_type, image_name=image_name, image=image[0]
        )



