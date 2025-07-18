import os
import math
import argparse
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
import torch
from transformers import (
    CLIPModel,
    CLIPProcessor,
    AutoModel,
    AutoImageProcessor,
)
from torchvision import transforms
# from torchmetrics.image.fid import FrechetInceptionDistance
# (改) 改回下面的注释
# from .GraphAdapter import PadToSquare
# from .infer_pipeline import InstructG2IPipeline
from GraphAdapter import PadToSquare
from infer_pipeline import InstructG2IPipeline
import wandb

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

parser = argparse.ArgumentParser(description="Run inference with InstructG2I.")
# (改) 改下面路径
# parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
parser.add_argument("--config", 
                    type=str, 
                    default='./config/test_Movies.json',
                    help="Path to the config file.")
args = parser.parse_args()

config = load_config(args.config)
args = argparse.Namespace(**config)

def read_data(test_dir):
    data = []
    with open(os.path.join(test_dir, 'metadata.jsonl')) as f:
        readin = f.readlines()
        for line in tqdm(readin):
            tmp = json.loads(line)
            # data.append({
            #     'text': tmp['text'],
            #     'center_image': Image.open(os.path.join(test_dir, tmp['center'])).convert("RGB"),
            #     'neighbor_image': [Image.open(os.path.join(test_dir, fname)).convert("RGB") for fname in tmp[args.neighbor_key]]
            # })
            data.append({
                'text': tmp['text'],
                'center_image': tmp['center'],
                'neighbor_image': tmp[args.neighbor_key]
            })
    return data

# Evaluator
# clip_id = "openai/clip-vit-large-patch14"
# dino_id = "facebook/dinov2-large"
clip_id = os.path.join(args.cache_dir,"clip-vit-large-patch14")
dino_id = os.path.join(args.cache_dir,"dinov2-large")
clip_model = CLIPModel.from_pretrained(clip_id, cache_dir=args.cache_dir).to(args.device)
clip_processor = CLIPProcessor.from_pretrained(clip_id, cache_dir=args.cache_dir)
dino_model = AutoModel.from_pretrained(dino_id, cache_dir=args.cache_dir).to(args.device)
dino_processor = AutoImageProcessor.from_pretrained(dino_id, cache_dir=args.cache_dir)


def main():
    if args.if_wandb:
        wandb.init(
            project="Instructg2i",
            name="{}".format(args.wandb_init_name),
            mode="online",
            config=args
        )

    print(args.neighbor_key)

    # read the data(读metadata.jsonl)
    print('Reading data...')
    dataset = read_data(args.test_dir)

    # image transformation function
    neighbor_transforms = transforms.Compose(
                [
                    PadToSquare(fill=(args.resolution, args.resolution, args.resolution), padding_mode='constant'),
                    transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(args.resolution)
                ]
        )

    def neighbor_transform_func(neighbor_images, gt_image):
        neighbor_image = [neighbor_transforms(n_img) for n_img in neighbor_images]
        neighbor_image += [neighbor_transforms(Image.fromarray(np.uint8(np.zeros_like(np.array(gt_image)))).convert('RGB'))] * (args.neighbor_num - len(neighbor_image))
        return neighbor_image
    
    def neighbor_transform_func_mine(neighbor_images):
        neighbor_images += ['Empty'] * (args.neighbor_num - len(neighbor_images))
        return neighbor_images

    def neighbor_mask_func(neighbor_images):
        neighbor_mask = [1] * len(neighbor_images)
        neighbor_mask += [0] * (args.neighbor_num - len(neighbor_mask))
        return neighbor_mask

    # init the pipeline(加载模型！最重要一步)
    print('Loading diffusion model...')
    pipe_graph2img = InstructG2IPipeline.from_pretrained(args.model_dir, args.neighbor_num, device=args.device,args=args)

    # run inference
    print('Scoring...')
    img_clip_scores = []
    dinov2_scores = []

    print(f'Total testing data:{len(dataset)}, max index: {args.max_index}')
    assert args.max_index <= len(dataset)
    num_diff_iter = math.ceil(args.max_index / args.diffusion_infer_batch_size)
    num_score_iter = math.ceil(args.max_index / args.score_batch_size)

    # diffusion model inference
    gt_images = []
    gen_images = []
    for idx in tqdm(range(num_diff_iter), desc="Diffusion model inference"):
        start = idx * args.diffusion_infer_batch_size
        end = min(args.max_index, (idx + 1) * args.diffusion_infer_batch_size)

        # get current batch data
        texts = [dataset[idd]['text'] for idd in range(start, end)]
        # neighbor_images = [neighbor_transform_func(dataset[idd]["neighbor_image"][:args.neighbor_num], dataset[idd]["center_image"]) for idd in range(start, end)]
        # 此时neighbor_images已经成为了node_id，而非Image.open类型
        neighbor_images = [neighbor_transform_func_mine(dataset[idd]["neighbor_image"][:args.neighbor_num]) for idd in range(start, end)]

        # 其实并不是mask，没有遮盖住实际存在的邻居，而是确定每一个位置是否有邻居，为存在邻居标记1，不存在标记0，总长度为args.neighbor_num
        neighbor_masks = [neighbor_mask_func(dataset[idd]["neighbor_image"][:args.neighbor_num]) for idd in range(start, end)]
                
        gen_image = pipe_graph2img(prompt=texts, neighbor_image=neighbor_images, neighbor_mask=torch.LongTensor(neighbor_masks), num_inference_steps=args.num_inference_steps).images
        if args.if_wandb:
            wandb.log({
                "example_image": wandb.Image(gen_image)
                })
        if args.if_save_image:
            cnt = 0
            os.makedirs(args.save_image_dir, exist_ok=True)
            for example_image in gen_image:
                example_image.save(os.path.join(args.save_image_dir,f"example{cnt}.jpg"))
                cnt += 1
        # prepare for later score calculation
        gt_images.extend([dataset[idd]["center_image"] for idd in range(start, end)])
        gen_images.extend(gen_image)

    for idx in tqdm(range(num_score_iter), desc="Score model inference"):
        start = idx * args.score_batch_size
        end = min(args.max_index, (idx + 1) * args.score_batch_size) 

        # clip scores
        gt_image_dp = clip_processor(images=[gt_images[idd] for idd in range(start, end)], return_tensors="pt", padding=True)
        gen_image_dp = clip_processor(images=[gen_images[idd] for idd in range(start, end)], return_tensors="pt", padding=True)

        gt_image_dp = {k: v.to(args.device) for k, v in gt_image_dp.items()}
        gen_image_dp = {k: v.to(args.device) for k, v in gen_image_dp.items()}
        
        with torch.no_grad():
            gt_image_features = clip_model.get_image_features(**gt_image_dp)
            gen_image_features = clip_model.get_image_features(**gen_image_dp)

            gt_image_features = gt_image_features / gt_image_features.norm(p=2, dim=1, keepdim=True)
            gen_image_features = gen_image_features / gen_image_features.norm(p=2, dim=1, keepdim=True)
            
            img_clip_score = torch.nn.functional.relu(torch.diagonal(torch.matmul(gt_image_features, gen_image_features.t()), 0))
                    
            img_clip_scores.extend(img_clip_score.tolist())

        # dino-v2 score
        gt_image_dp_dino = dino_processor(images=[gt_images[idd] for idd in range(start, end)], return_tensors="pt")
        gen_image_dp_dino = dino_processor(images=[gen_images[idd] for idd in range(start, end)], return_tensors="pt")

        gt_image_dp_dino = {k: v.to(args.device) for k, v in gt_image_dp_dino.items()}
        gen_image_dp_dino = {k: v.to(args.device) for k, v in gen_image_dp_dino.items()}

        with torch.no_grad():
            gt_image_features_dino = dino_model(**gt_image_dp_dino).pooler_output
            gen_image_features_dino = dino_model(**gen_image_dp_dino).pooler_output

            gt_image_features_dino = gt_image_features_dino / gt_image_features_dino.norm(p=2, dim=1, keepdim=True)
            gen_image_features_dino = gen_image_features_dino / gen_image_features_dino.norm(p=2, dim=1, keepdim=True)

            dino_score = torch.nn.functional.relu(torch.diagonal(torch.matmul(gt_image_features_dino, gen_image_features_dino.t()), 0))
            dinov2_scores.extend(dino_score.tolist())

    # fid = FrechetInceptionDistance()
    # fid_transforms = transforms.Compose(
    #         [
    #             PadToSquare(fill=(args.resolution, args.resolution, args.resolution), padding_mode='constant'),
    #             transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
    #             transforms.CenterCrop(args.resolution),
    #             transforms.ToTensor(),
    #             transforms.ConvertImageDtype(torch.uint8)
    #         ]
    # )
    # gt_images_tensor = torch.stack([fid_transforms(tmp_img) for tmp_img in gt_images])
    # gen_images_tensor = torch.stack([fid_transforms(tmp_img) for tmp_img in gen_images])
    # fid.update(gt_images_tensor, real=True)
    # fid.update(gen_images_tensor, real=False)
    # fid_score = fid.compute().item()

    print('**************************** Ground Truth-based Metrics **************************************')
    print(f"Generated Image v.s. Ground Truth Image CLIP score: {np.mean(img_clip_scores)}")
    print(f"Generated Image v.s. Ground Truth Image Dino-v2 score: {np.mean(dinov2_scores)}")
    # print(f"Generated Image v.s. Ground Truth Image FiD score: {fid_score}")
    if args.if_wandb:
        wandb.finish()

if __name__=='__main__':
    main()

        