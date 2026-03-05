"""
FlashHead WebUI - Gradio interface for SoulX-FlashHead
Upload a portrait image + audio → generate talking head video
"""
import os
import sys

# Disable torch.compile/dynamo to avoid sageattn CUDA errors
os.environ["TORCHDYNAMO_DISABLE"] = "1"


import time
import tempfile
import subprocess
import shutil
from datetime import datetime

import gradio as gr
import numpy as np
import imageio
import librosa
import torch
from loguru import logger

# Add project root to path
PROJECT_ROOT = os.path.expanduser("~/SoulX-FlashHead")
sys.path.insert(0, PROJECT_ROOT)

from flash_head.inference import get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline

# Global pipeline (loaded once)
PIPELINE = None
MODEL_TYPE = "lite"
CKPT_DIR = os.path.join(PROJECT_ROOT, "models/SoulX-FlashHead-1_3B")
WAV2VEC_DIR = os.path.join(PROJECT_ROOT, "models/wav2vec2-base-960h")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "webui_outputs")


def load_pipeline():
    global PIPELINE
    if PIPELINE is None:
        logger.info("Loading FlashHead pipeline (this may take a moment)...")
        PIPELINE = get_pipeline(world_size=1, ckpt_dir=CKPT_DIR, wav2vec_dir=WAV2VEC_DIR, model_type=MODEL_TYPE)
        logger.info("Pipeline loaded!")
    return PIPELINE


def save_video(frames_list, video_path, audio_path, fps):
    temp_video_path = video_path.replace('.mp4', '_tmp.mp4')
    with imageio.get_writer(temp_video_path, format='mp4', mode='I',
                            fps=fps, codec='h264', ffmpeg_params=['-bf', '0']) as writer:
        for frames in frames_list:
            frames = frames.numpy().astype(np.uint8)
            for i in range(frames.shape[0]):
                writer.append_data(frames[i, :, :, :])
    
    cmd = ['ffmpeg', '-i', temp_video_path, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', '-shortest', video_path, '-y']
    subprocess.run(cmd, capture_output=True)
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)


def generate_video(image_path, audio_path, model_type="lite", seed=42, use_face_crop=False, progress=gr.Progress()):
    if image_path is None or audio_path is None:
        return None, "⚠️ Please upload both an image and an audio file."
    
    global MODEL_TYPE, PIPELINE
    
    # Reload pipeline if model type changed - free old model first
    if model_type != MODEL_TYPE:
        MODEL_TYPE = model_type
        if PIPELINE is not None:
            logger.info(f"Switching model to {model_type}, freeing old pipeline...")
            # Clear all tensors in pipeline
            del PIPELINE
            PIPELINE = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info(f"GPU memory freed. VRAM used: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    
    progress(0.1, desc="Loading model...")
    pipeline = load_pipeline()
    
    progress(0.2, desc="Preparing image...")
    get_base_data(pipeline, cond_image_path_or_dir=image_path, base_seed=seed, use_face_crop=use_face_crop)
    infer_params = get_infer_params()
    
    sample_rate = infer_params['sample_rate']
    tgt_fps = infer_params['tgt_fps']
    cached_audio_duration = infer_params['cached_audio_duration']
    frame_num = infer_params['frame_num']
    motion_frames_num = infer_params['motion_frames_num']
    slice_len = frame_num - motion_frames_num
    
    progress(0.3, desc="Loading audio...")
    human_speech_array_all, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    audio_duration = len(human_speech_array_all) / sample_rate
    
    progress(0.4, desc="Generating video chunks...")
    generated_list = []
    
    # Stream mode
    cached_audio_length_sum = sample_rate * cached_audio_duration
    audio_end_idx = cached_audio_duration * tgt_fps
    audio_start_idx = audio_end_idx - frame_num
    
    from collections import deque
    audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)
    
    human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
    num_slices = len(human_speech_array_all) // human_speech_array_slice_len
    human_speech_array_slices = human_speech_array_all[:num_slices * human_speech_array_slice_len].reshape(-1, human_speech_array_slice_len)
    
    total_chunks = len(human_speech_array_slices)
    start_time = time.time()
    
    for chunk_idx, human_speech_array in enumerate(human_speech_array_slices):
        torch.cuda.synchronize()
        chunk_start = time.time()
        
        audio_dq.extend(human_speech_array.tolist())
        audio_array = np.array(audio_dq)
        audio_embedding = get_audio_embedding(pipeline, audio_array, audio_start_idx, audio_end_idx)
        video = run_pipeline(pipeline, audio_embedding)
        
        torch.cuda.synchronize()
        chunk_time = time.time() - chunk_start
        
        generated_list.append(video.cpu())
        prog = 0.4 + 0.5 * (chunk_idx + 1) / total_chunks
        progress(prog, desc=f"Chunk {chunk_idx+1}/{total_chunks} ({chunk_time:.2f}s)")
    
    total_time = time.time() - start_time
    
    progress(0.95, desc="Saving video...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"flashhead_{timestamp}.mp4")
    save_video(generated_list, output_path, audio_path, fps=tgt_fps)
    
    avg_fps = total_chunks * slice_len / total_time if total_time > 0 else 0
    status = (
        f"✅ Done! {total_chunks} chunks in {total_time:.1f}s\n"
        f"📊 Avg speed: {avg_fps:.1f} FPS | Audio: {audio_duration:.1f}s\n"
        f"💾 Saved: {output_path}"
    )
    
    return output_path, status


# Build UI
with gr.Blocks(title="FlashHead WebUI") as demo:
    gr.Markdown("# 🎭 SoulX-FlashHead WebUI\nUpload a portrait image + audio → generate talking head video")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Portrait Image (PNG/JPG)")
            audio_input = gr.Audio(type="filepath", label="Audio (WAV/MP3)")
            
            with gr.Row():
                model_choice = gr.Radio(
                    choices=["lite", "pro"],
                    value="lite",
                    label="Model",
                    info="Lite: fast (96 FPS on 4090) | Pro: high quality (10.8 FPS)"
                )
                seed_input = gr.Number(value=42, label="Seed", precision=0)
            
            face_crop = gr.Checkbox(value=False, label="Auto face crop")
            generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            video_output = gr.Video(label="Generated Video")
            status_output = gr.Textbox(label="Status", lines=3)
    
    gr.Markdown("### 📂 Examples")
    gr.Examples(
        examples=[
            [os.path.join(PROJECT_ROOT, "examples/girl.png"), 
             os.path.join(PROJECT_ROOT, "examples/podcast_sichuan_16k.wav"),
             "lite", 42, False],
        ],
        inputs=[image_input, audio_input, model_choice, seed_input, face_crop],
        outputs=[video_output, status_output],
        fn=generate_video,
        cache_examples=False,
    )
    
    generate_btn.click(
        fn=generate_video,
        inputs=[image_input, audio_input, model_choice, seed_input, face_crop],
        outputs=[video_output, status_output],
    )

if __name__ == "__main__":
    # Preload pipeline on startup
    print("Pre-loading FlashHead pipeline...")
    load_pipeline()
    print("Ready! Launching WebUI...")
    demo.launch(server_name="0.0.0.0", server_port=8108, share=False, theme=gr.themes.Soft())
