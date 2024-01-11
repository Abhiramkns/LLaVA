import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from transformers.generation.streamers import TextIteratorStreamer
from torch.utils.data import DataLoader, Dataset, TensorDataset 

from PIL import Image

import requests
from io import BytesIO

from cog import BasePredictor, Input, Path, ConcatenateIterator
import time
import subprocess
from threading import Thread
import json
from tqdm import tqdm
from pathlib import Path
import random
from argparse import ArgumentParser
import os
os.environ["HUGGINGFACE_HUB_CACHE"] = os.getcwd() + "/weights"

# url for the weights mirror
REPLICATE_WEIGHTS_URL = "https://weights.replicate.delivery/default"
# files to download from the weights mirrors
weights = [
    {
        "dest": "liuhaotian/llava-v1.5-13b",
        # git commit hash from huggingface
        "src": "llava-v1.5-13b/006818fc465ebda4c003c0998674d9141d8d95f8",
        "files": [
            "config.json",
            "generation_config.json",
            "pytorch_model-00001-of-00003.bin",
            "pytorch_model-00002-of-00003.bin",
            "pytorch_model-00003-of-00003.bin",
            "pytorch_model.bin.index.json",
            "special_tokens_map.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ]
    },
    {
        "dest": "openai/clip-vit-large-patch14-336",
        "src": "clip-vit-large-patch14-336/ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
        "files": [
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin"
        ],
    }
]

class LaionDatasetShard(Dataset):
    def __init__(self, folder, image_processor, tokenizer, start_index, end_index):
        self.start = start_index
        self.end = end_index
        self.img_processor = image_processor
        self.tokenizer = tokenizer
        path = Path(folder)
        text_files = [*path.glob('**/*.txt')]
        image_files = [
            *path.glob('**/*.png'), *path.glob('**/*.jpg'),
            *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
        ]

        text_files = {text_file.stem: text_file for text_file in text_files}
        image_files = {image_file.stem: image_file for image_file in image_files}
        keys = (image_files.keys() & text_files.keys())

        self.keys = list(keys)
        self.keys.sort()
        self.keys = self.keys[start_index: end_index]
        self.text_files = {k: v for k, v in text_files.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}
        
    def __len__(self):
        return len(self.keys)
        
    def __getitem__(self, index):
        key = self.keys[index]
        with open(self.text_files[key], 'r') as f:
            caption = f.read()
            
        return load_image(str(self.image_files[key])), caption

    def collate_fn(self, dataset):
        images = []
        input_ids = []
        attn_mask = []
        for data in dataset:
            img_tensor = self.img_processor.preprocess(data[0], return_tensors="pt").pixel_values.half()
            images.append(img_tensor.squeeze(0))
            # inputs = self.tokenizer(data[1], return_tensors="pt", max_length=256, padding="max_length")
            # input_ids.append(inputs.input_ids)
            # attn_mask.append(inputs.attention_mask)
        images = torch.stack(images)
        # input_ids = torch.stack(input_ids)
        # attn_mask = torch.stack(attn_mask)

        return images #, input_ids, attn_mask
    
def main(args):
    p = Predictor()
    for weight in weights:
        download_weights(weight["src"], weight["dest"], weight["files"])
    disable_torch_init()
    tokenizer, model, image_processor, context_len = load_pretrained_model("liuhaotian/llava-v1.5-13b-", model_name="llava-v1.5-13b", model_base=None, load_8bit=False, load_4bit=False)
    path = args.dataset_path
    dataset = LaionDatasetShard(path, image_processor, tokenizer, args.start_index, args.end_index)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn)
    p.predict(dataloader, model, tokenizer, context_len, start_index=args.start_index, end_index=args.end_index, output_folder=args.output_folder)
    print("Done")
    
def download_json(url: str, dest: Path):
    res = requests.get(url, allow_redirects=True)
    if res.status_code == 200 and res.content:
        with dest.open("wb") as f:
            f.write(res.content)
    else:
        print(f"Failed to download {url}. Status code: {res.status_code}")

def download_weights(baseurl: str, basedest: str, files: list[str]):
    basedest = Path(basedest)
    start = time.time()
    print("downloading to: ", basedest)
    basedest.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = basedest / f
        url = os.path.join(REPLICATE_WEIGHTS_URL, baseurl, f)
        if not dest.exists():
            print("downloading url: ", url)
            if dest.suffix == ".json":
                download_json(url, dest)
            else:
                subprocess.check_call(["pget", url, str(dest)], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def predict(
        self,
        # image: Path = Input(description="Input image"),
        # prompt: str = Input(description="Prompt to use for text generation"),
        # top_p: float = Input(description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens", ge=0.0, le=1.0, default=1.0),
        # temperature: float = Input(description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic", default=0.2, ge=0.0),
        # max_tokens: int = Input(description="Maximum number of tokens to generate. A word is generally 2-3 tokens", default=1024, ge=0),
        train_dataloader,
        model, 
        tokenizer,
        context_len,
        top_p = 1.0,
        temperature = 0.2,
        max_tokens = 1024,
        start_index=0,
        end_index=0,
        output_folder='.'
    ):
        """Run a single prediction on the model"""

        prompt = """Can you please describe this image in up to two paragraphs? Please specify any objects within the image, backgrounds, scenery, interactions, and gestures or poses. If they are multiple of any object, please specify how many. Is there text in the image, and if so, what does it say? If there is any lighting in the image, can you identify where it is and what it looks like? What style is the image? If there are people or characters in the image, what emotions are they conveying? Please keep your descriptions factual and terse but complete. DO NOT add any unnecessary speculation about the things that are not part of the image such as "the image is inspiring to viewers" or "seeing this makes you feel joy". DO NOT add things such as "creates a unique and entertaining visual", as these descriptions are interpretations and not a part of the image itself. The description should be purely factual, with no subjective speculation. Make sure to include the style of the image, for example cartoon, photograph, 3d render etc. Start with the words This image showcases:"""
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()

        # loop start
    
        # just one turn, always prepend image token
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        outputs = []
        start_time = time.time()
        dic = {}
        for batch in tqdm(train_dataloader):
            batch =batch.cuda()
            # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            # keywords = [stop_str]
            # stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            # streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=20.0)
            input_ids = [tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()]*batch.shape[0]
            input_ids = torch.stack(input_ids)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            
            with torch.inference_mode():
                output_ids = model.generate(
                    inputs=input_ids,
                    images=batch,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )
    
                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
                for i, o in enumerate(outputs):
                    key = start_index+i
                    dic[key] = o
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time elapsed: {elapsed_time} seconds")
        with open(f'{output_folder}/llava-captions{start_index}-{end_index}.json', 'w') as f:
            json.dump(dic, f)

            # thread = Thread(target=self.model.generate, kwargs=dict(
            #     inputs=input_ids,
            #     images=image_tensor,
            #     do_sample=True,
            #     temperature=temperature,
            #     top_p=top_p,
            #     max_new_tokens=max_tokens,
            #     streamer=streamer,
            #     use_cache=True,
            #     stopping_criteria=[stopping_criteria]))
            # thread.start()
            # workaround: second-to-last token is always " "
            # but we want to keep it if it's not the second-to-last token
            # prepend_space = False
            # for new_text in streamer:
            #     if new_text == " ":
            #         prepend_space = True
            #         continue
            #     if new_text.endswith(stop_str):
            #         new_text = new_text[:-len(stop_str)].strip()
            #         prepend_space = False
            #     elif prepend_space:
            #         new_text = " " + new_text
            #         prepend_space = False
            #     if len(new_text):
            #         return new_text
            # if prepend_space:
            #     return " "
            # thread.join()
        
def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()

    parser.add_argument('--start_index', type=int, help='A positional argument.', default=0)
    parser.add_argument('--end_index',  type=int, help='An optional argument.', default=1)
    parser.add_argument('--dataset_path', type=str, default='/scratch/nkusumba/laion-dataset/dataset')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_folder', type=str, default='.')
    parser.add_argument('--model_ckpt', type=str, default='.')
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()
    main(args)
    
