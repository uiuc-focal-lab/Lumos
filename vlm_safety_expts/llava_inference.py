import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

# llava-hf/llava-1.5-7b-hf

class LlavaInference:
    def __init__(self):
        # Load the model in half-precision
        self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    def load_image(self, image_path):
        return Image.open(image_path).convert("RGB")

    def infer(self, image_path, text_prompt):
        image = self.load_image(image_path)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": f"{image_path}"},
                    {"type": "text", "text": f"{text_prompt}"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, torch.float16)

        generate_ids = self.model.generate(**inputs, max_new_tokens=30)
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True)

        # Extract response from output
        return output[0].split("ASSISTANT: ")[1]

# llava = LlavaInference()
# response = llava.infer("scene.png", "question")
# print(response)