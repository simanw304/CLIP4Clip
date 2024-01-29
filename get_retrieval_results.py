import torch
from PIL import Image
import uuid
import os
import numpy as np
from tqdm import tqdm
import time
import json
from pathlib import Path
import tempfile
import csv


from PIL import Image
import cv2

# combine features to less files
batch_image_features = []
batch_media_ids = []
media_ids = []
batch_num = 0
processed = []

output_path = '/nfs/swang7/500k_db/l14_features'
features = [os.path.join(output_path, feature) for feature in os.listdir(output_path)]

for feature in tqdm(features):
    data = np.load(feature) # (N, 768)
    features = torch.from_numpy(data['features'])

    batch_image_features.append(features) 
    batch_media_ids.append(data['media_ids'])

image_features = np.concatenate(batch_image_features, axis=0)
media_ids = np.concatenate(batch_media_ids, axis=0)
# # with open('processed_media_ids_336.txt', 'w') as fp:
# #     for i in media_ids:
# #         fp.write(f'{i}\n')
# print(image_features.shape)
# print(len(media_ids))
np.savez(f'/nfs/swang7/500k_db/all_features/l14_features.npz', media_ids=media_ids, features=image_features)

import os
import sys
from tqdm import tqdm

scriptpath = "/nfs/swang7/dev/CLIP4Clip/"

sys.path.append(os.path.abspath(scriptpath))

import uuid
import torch
import numpy as np
import random
import os
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import time
import argparse
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip
from modules.optimization import BertAdam

from torch.utils.data import Dataset
import pandas as pd
from dataloaders.rawvideo_util import RawVideoExtractor
from util import parallel_apply, get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT

SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}


def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", type=str, default='ckpts/ckpt_msrvtt_retrieval_looseType',
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default='/nfs/swang7/dev/CLIP4Clip/ckpts/ckpt_spotlight_retrieval_l14_31k_0122/pytorch_model.bin.9', type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")
    parser.add_argument("--input_list", type=str, help="")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', action='store_true', help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")

    parser.add_argument("--pretrained_clip_name", default="ViT-L/14", type=str, help="Choose a CLIP version")

    args = parser.parse_args(['--do_eval', '--loose_type'])

    if args.sim_header == "tightTransf":
        args.loose_type = False

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def init_model(args, device):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    model.to(device)

    return model
        
def main():
    
    args = get_args()
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu", args.local_rank)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = init_model(args, device)

    # model_state_dict = None
    # model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=args.cache_dir, state_dict=model_state_dict, task_config=args)
    # model.to(device)
    model.eval()

    # data = np.load('/nfs/swang7/500k_db/all_features/336_features.npz')
    # features = data['features']
    # media_ids = data['media_ids']


    with open('/nfs/swang7/500k_db/metadata/556_queries.txt', 'r') as fp:
        data = fp.readlines()
    queries = [i.strip() for i in data]

    composite_to_submission = {}
    submission_to_composite = {}
    with open('/nfs/swang7/500k_db/metadata/500k_spotlight_db_meta.ndjson', 'r') as fp:
        for line in fp:
            data = json.loads(line)
            composite_id = data['composite_story_id']
            composite_to_submission[composite_id] = []
            for multi_snap_meta in data['multi_snap_meta']:
                submission_id = multi_snap_meta['submission_id']
                composite_to_submission[composite_id].append(submission_id)
                submission_to_composite[submission_id] = composite_id

    text_inputs = []
    tokenizer = ClipTokenizer()
    max_words = 30

    for query in queries:
        pairs_text = np.zeros((1, max_words), dtype=np.int32)
        words = tokenizer.tokenize(query)

        words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]

        input_ids = tokenizer.convert_tokens_to_ids(words)
        while len(input_ids) < max_words:
            input_ids.append(0)

        pairs_text[0] = np.array(input_ids)
        input_ids = torch.tensor(pairs_text).to(device)
        with torch.no_grad():
            text_features = model.clip.encode_text(input_ids)
        text_inputs.append(text_features)

    sequence_output = torch.cat(text_inputs, axis=0).float().to(device)
    visual_output = torch.tensor(image_features).float().to(device)

    video_mask = np.ones((1, 10), dtype=np.long)
    video_mask = torch.as_tensor(video_mask).float().to(device)

    visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
    visual_output = model._mean_pooling_for_similarity_visual(visual_output, video_mask)
    visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

    sequence_output = sequence_output.squeeze(1)
    sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
    
    logit_scale = model.clip.logit_scale.exp().float()
    retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
    sim_matrix = retrieve_logits.detach().cpu().numpy()

    print(sim_matrix.shape)


    def find_top_n_idx(x, n):
        return x.argsort()[-n:][::-1]

    import subprocess

    def check_video_format(file_path):
        # Run ffprobe to get video information
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)

        # Check if the codec is H.264 or H.265 (HEVC)
        codec_name = output.strip()
        return codec_name

    def transcode_to_h264(input_file, output_file):
        # Run FFmpeg to transcode the video to H.264
        cmd = ['ffmpeg', '-i', input_file, '-c:v', 'libx264', '-crf', '23', '-c:a', 'aac', output_file]
        subprocess.check_call(cmd)

        return "Transcoding successful"

    from google.cloud import storage
    client = storage.Client()
    # Specify your GCS bucket name and file path
    bucket_name = 'multimodal-ai'
    # Get the specified GCS bucket
    # bucket = client.get_bucket(bucket_name)
    bucket = client.bucket(bucket_name)

    def check_file_exists(video_id):
        """
        Check if a file exists in Google Cloud Storage.

        :param bucket_name: The name of the GCS bucket.
        :param file_path: The path to the file in the bucket.
        :return: True if the file exists, False otherwise.
        """

        # Check if the file exists in the bucket
        blob = bucket.blob(f't2v_benchmark/500k_retrieval_results/{video_id}.mp4')
        return blob.exists()

    top_n = 50
    gcs_dir = 'gs://multimodal-ai/t2v_benchmark/500k_retrieval_results'
    output = []

    for i, query in tqdm(enumerate(queries), total=len(queries), desc="Processing"):
        rank = 0
        seen_story = set()
        top_n_idx = find_top_n_idx(sim_matrix[i], top_n)    
        video_ids = [str(media_ids[idx]) for idx in top_n_idx]
        scores = [sim_matrix[i][idx] for idx in top_n_idx]
        
        for video_id_idx, video_id in enumerate(video_ids):
            
            # some blip2 embeddings are from WA/TX/IL where the metadata table doesn't include
            # if video_id not in submission_to_composite:
            #     continue
            
            # # skip seen composite video id
            # composite_id = submission_to_composite[video_id]
            # if composite_id in seen_story:
            #     continue
            
            # # will skip uploading if the video exists in our gcs dir
            # if not check_file_exists(video_id):
            #     # transcode the submission video if needed, upload to gcs
            #     submission_video_deleted = False
            #     with tempfile.TemporaryDirectory() as temp_dir:
            #         temp_file = os.path.join(temp_dir, "tempfile.mp4")
            #         temp_file_transcoded = os.path.join(temp_dir, "temp_file_transcoded.mp4")
            #         try:
            #             os.system(f'gsutil cp gs://ourstorymedia/{video_id}.mp4 {temp_file}')
            #             # Check the video format
            #             format_result = check_video_format(temp_file)
            #             if format_result == "hevc":
            #                 # Video is in HEVC format, so transcode it to H.264
            #                 transcode_result = transcode_to_h264(temp_file, temp_file_transcoded)
            #                 os.system(f'gsutil cp {temp_file_transcoded} {gcs_dir}/{video_id}.mp4')
            #             elif format_result == "h264":
            #                 os.system(f'gsutil cp {temp_file} {gcs_dir}/{video_id}.mp4')
            #             else:
            #                 print("Unsupported video codec")
            #                 submission_video_deleted = True
            #         except Exception as e:
            #             # print(e)
            #             submission_video_deleted = True
        
            #     if submission_video_deleted:
            #             continue
                    
            # seen_story.add(composite_id)
            output.append({'query': query, 'rank': rank, 'submission_id': video_id, 'score': str(scores[video_id_idx])}) # 'composite_id': composite_id,
            rank += 1
            
            if rank >= 6:
                break

    file_path = "output_l14_500k_top50"

    # Open the file for writing
    with open(f'{file_path}.ndjson', 'w') as file:
        # Iterate over each dictionary in the data list
        for item in output:
            # Serialize the dictionary to a JSON string and write it to the file with a newline
            json.dump(item, file)
            file.write('\n')

    fieldnames = ["text_example", "video_url"]

    # Open the CSV file for writing
    with open(f'{file_path}.csv', mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(fieldnames)

        # Write the data rows
        for row in output:
            writer.writerow([row['query'], f'gs://multimodal-ai/t2v_benchmark/500k_retrieval_results/{row["submission_id"]}.mp4'])

if __name__ == "__main__":
    main()
