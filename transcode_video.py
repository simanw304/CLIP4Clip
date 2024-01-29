import csv
import json
from tqdm import tqdm
import google.cloud.storage
import tempfile
import os

# with open('/nfs/swang7/500k_db/metadata/556_queries.txt', 'r') as fp:
#     data = fp.readlines()
# queries = [i.strip() for i in data]

metadata = {}
with open('output_l14_500k_top50.ndjson', 'r') as fp:
    lines = fp.readlines()
    for i in lines:
        data = json.loads(i)
        if data['query'] not in metadata:
            metadata[data['query']] = {'submission_id': [0] * 50, 'score': [0] * 50}
        rank = data['rank']
        metadata[data['query']]['submission_id'][rank] = data['submission_id']
        metadata[data['query']]['score'][rank] = data['score']

def find_top_n_idx(x, n):
    return x.argsort()[-n:][::-1]

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

for i, query in tqdm(enumerate(metadata), total=len(metadata), desc="Processing"):
    rank = 0
    seen_story = set()

    video_ids = metadata[query]['submission_id']
    scores = metadata[query]['score']
    
    for video_id_idx, video_id in enumerate(video_ids):
        
        # some blip2 embeddings are from WA/TX/IL where the metadata table doesn't include
        if video_id not in submission_to_composite:
            continue
        
        # skip seen composite video id
        composite_id = submission_to_composite[video_id]
        if composite_id in seen_story:
            continue
        
        # will skip uploading if the video exists in our gcs dir
        if not check_file_exists(video_id):
            # transcode the submission video if needed, upload to gcs
            submission_video_deleted = False
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = os.path.join(temp_dir, "tempfile.mp4")
                temp_file_transcoded = os.path.join(temp_dir, "temp_file_transcoded.mp4")
                try:
                    os.system(f'gsutil cp gs://ourstorymedia/{video_id}.mp4 {temp_file}')
                    # Check the video format
                    format_result = check_video_format(temp_file)
                    if format_result == "hevc":
                        # Video is in HEVC format, so transcode it to H.264
                        transcode_result = transcode_to_h264(temp_file, temp_file_transcoded)
                        os.system(f'gsutil cp {temp_file_transcoded} {gcs_dir}/{video_id}.mp4')
                    elif format_result == "h264":
                        os.system(f'gsutil cp {temp_file} {gcs_dir}/{video_id}.mp4')
                    else:
                        print("Unsupported video codec")
                        submission_video_deleted = True
                except Exception as e:
                    # print(e)
                    submission_video_deleted = True
    
            if submission_video_deleted:
                    continue
                
        seen_story.add(composite_id)
        output.append({'query': query, 'rank': rank, 'submission_id': video_id, 'score': str(scores[video_id_idx])}) # 'composite_id': composite_id,
        rank += 1
        
        if rank >= 6:
            break

file_path = "output_l14_500k_top6"

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
