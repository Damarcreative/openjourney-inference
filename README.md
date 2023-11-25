# openjourney-inference

## Description
This is a Python script to generate images using the diffusers library with OpenJourney models with various types of schedulers. This script uses Gradio to create a simple user interface for setting parameters and viewing the resulting images.

<table class="custom-table">
  <tr>
    <td>
      <a href="https://huggingface.co/Linaqruf/animagine-xl/blob/main/sample_images/image (1).png">
        <img class="custom-image" src="image/image (1).png" alt="sample1">
      </a>
      <a href="https://huggingface.co/Linaqruf/animagine-xl/blob/main/sample_images/image (3).png">
        <img class="custom-image" src="image/image (3).png" alt="sample3">
      </a>
    </td>
    <td>
      <a href="https://huggingface.co/Linaqruf/animagine-xl/blob/main/sample_images/image (2).png">
        <img class="custom-image" src="image/image (2).png" alt="sample2">
      </a>
      <a href="https://huggingface.co/Linaqruf/animagine-xl/blob/main/sample_images/image (4).png">
        <img class="custom-image" src="image/image (4).png" alt="sample4">
      </a>
    </td>
  </tr>
</table>

## Usage Steps
1. Install dependencies by running the following command:
```
pip install -q diffusers transformers omegaconf accelerate gradio
```
2. Run the script and enter the prompt and other parameters via the user interface that appears.

3. Select the scheduler type from the dropdown provided.

4. Determine the location to save the resulting image by setting folder_path in the markdown section.

5. Press the generate button to generate an image.

### Parameter
`scheduler_type`: Scheduler type for the diffusion model.

`prompt`: Prompt to generate image.

`negative_prompt`: Negative prompt to form an image concept.

`width`: The width of the resulting image.

`height`: The height of the resulting image.

`guidance_scale`: Guidance scale to control the degree to which guidance influences results.

`num_inference_steps`: Number of inference steps used.

### Usage Example
#### Import Libraries
```
import gradio as gr
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler
)
import torch
import os
import re
from PIL import Image
```

#### Load Model
```
repo_id = "prompthero/openjourney-v4"
pipe = StableDiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe.to("cuda")
```







