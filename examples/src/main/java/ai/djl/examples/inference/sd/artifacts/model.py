import io
import torch
from djl_python import Output
from minimum_pipe import MinDiffusionPipeline

trained_model_path = "./stable-diffusion-v1-4"


class TextStableDiffusionPipe:
    global trained_model_path

    def __init__(self, prompt, scale=7.5, steps=20, height=512, width=512):
        self.prompt = prompt
        self.scale = scale
        self.steps = steps
        self.height = height
        self.width = width
        self.pipeline = self.set_pipeline()

    def set_pipeline(self):
        if torch.cuda.is_available():
            return MinDiffusionPipeline.from_pretrained(trained_model_path, revision="fp16",
                                                        torch_dtype=torch.float16).to("cuda")
        else:
            return MinDiffusionPipeline.from_pretrained(trained_model_path)

    def generate_image(self):
        self.prompt = self.pipeline.prompt_to_tensor(self.prompt)
        images = self.pipeline(self.prompt, guidance_scale=self.scale, num_inference_steps=self.steps,
                               height=self.height, width=self.width)
        buf = io.BytesIO()
        img = images['images'][0]
        img.show()
        img.save(buf, format='PNG')
        return buf.getvalue()


def process_request(req):
    content_type = req['content-type']
    if content_type == 'application/json':
        obj = req.get_as_json()
        return TextStableDiffusionPipe(obj['prompt'], obj['scale'], obj['steps'],
                                       obj['height'], obj['width']).generate_image()
    else:
        prompt = req['data']
        return TextStableDiffusionPipe(prompt).generate_image()


def handle(inputs) -> Output:
    image = process_request(inputs)
    return Output().add(image).add_property("content-type", "image/png")


if __name__ == "__main__":
    request = {'content-type': 'text/string', 'data': "an astronaut riding a horse"}
    output = handle(request)
