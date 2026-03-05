"""
FlashHead Streaming API Service
WebSocket endpoint for real-time audio → video frame generation
"""
import os
import sys
import time
import asyncio
import base64
import json
from collections import deque
from io import BytesIO

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from loguru import logger
from PIL import Image

PROJECT_ROOT = os.path.expanduser("~/SoulX-FlashHead")
sys.path.insert(0, PROJECT_ROOT)

from flash_head.inference import get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline

# ─── Config ───
CKPT_DIR = os.path.join(PROJECT_ROOT, "models/SoulX-FlashHead-1_3B")
WAV2VEC_DIR = os.path.join(PROJECT_ROOT, "models/wav2vec2-base-960h")
MODEL_TYPE = "lite"
HOST = "0.0.0.0"
PORT = 8109

# ─── Global State ───
pipeline = None
infer_params = None

app = FastAPI(title="FlashHead Streaming API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def init_pipeline():
    """Load FlashHead pipeline once at startup."""
    global pipeline, infer_params
    logger.info(f"Loading FlashHead pipeline (model_type={MODEL_TYPE})...")
    pipeline = get_pipeline(world_size=1, ckpt_dir=CKPT_DIR, wav2vec_dir=WAV2VEC_DIR, model_type=MODEL_TYPE)
    infer_params = get_infer_params()
    logger.info("Pipeline loaded!")


def frames_to_jpeg_bytes(frames_tensor, quality=80):
    """Convert a batch of frames (N,H,W,3 uint8 numpy) to list of JPEG bytes."""
    frames_np = frames_tensor.numpy().astype(np.uint8)
    result = []
    for i in range(frames_np.shape[0]):
        img = Image.fromarray(frames_np[i])
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        result.append(buf.getvalue())
    return result


@app.on_event("startup")
async def startup():
    init_pipeline()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_type": MODEL_TYPE,
        "pipeline_loaded": pipeline is not None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_used_gb": round(torch.cuda.memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else 0,
    }


@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """
    Real-time streaming: receive audio chunks, return video frames.
    
    Protocol:
    1. Client sends JSON: {"action": "init", "image": "<base64 PNG/JPG>", "seed": 42}
       Server responds: {"status": "ready", "fps": 25, "frame_size": [512, 512]}
    
    2. Client sends binary audio chunks (PCM int16, 16kHz, mono)
       Server responds with binary JPEG frames prefixed by 4-byte frame count
    
    3. Client sends JSON: {"action": "stop"}
       Server closes connection.
    
    Audio chunk size should be: slice_len * sample_rate / tgt_fps bytes
    (roughly 0.5-1 second of audio per chunk)
    """
    await ws.accept()
    logger.info("WebSocket client connected")
    
    sample_rate = infer_params['sample_rate']
    tgt_fps = infer_params['tgt_fps']
    cached_audio_duration = infer_params['cached_audio_duration']
    frame_num = infer_params['frame_num']
    motion_frames_num = infer_params['motion_frames_num']
    slice_len = frame_num - motion_frames_num
    
    cached_audio_length_sum = sample_rate * cached_audio_duration
    audio_end_idx = cached_audio_duration * tgt_fps
    audio_start_idx = audio_end_idx - frame_num
    
    # Audio samples per chunk
    samples_per_chunk = slice_len * sample_rate // tgt_fps
    
    audio_dq = None  # initialized after "init"
    initialized = False
    chunk_idx = 0
    
    try:
        while True:
            data = await ws.receive()
            
            if "text" in data:
                msg = json.loads(data["text"])
                action = msg.get("action", "")
                
                if action == "init":
                    # Decode image
                    img_b64 = msg.get("image", "")
                    seed = msg.get("seed", 42)
                    
                    if not img_b64:
                        await ws.send_json({"error": "No image provided"})
                        continue
                    
                    # Save temp image
                    img_bytes = base64.b64decode(img_b64)
                    tmp_img = "/tmp/flashhead_ws_input.png"
                    with open(tmp_img, "wb") as f:
                        f.write(img_bytes)
                    
                    # Prepare pipeline
                    get_base_data(pipeline, cond_image_path_or_dir=tmp_img, base_seed=seed, use_face_crop=False)
                    
                    # Init audio buffer
                    audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)
                    chunk_idx = 0
                    initialized = True
                    
                    await ws.send_json({
                        "status": "ready",
                        "fps": tgt_fps,
                        "frame_size": [infer_params['height'], infer_params['width']],
                        "samples_per_chunk": samples_per_chunk,
                        "sample_rate": sample_rate,
                    })
                    logger.info(f"Session initialized (seed={seed}, samples_per_chunk={samples_per_chunk})")
                
                elif action == "stop":
                    logger.info("Client requested stop")
                    await ws.send_json({"status": "stopped"})
                    break
            
            elif "bytes" in data and initialized:
                # Binary audio data: PCM int16, 16kHz, mono
                audio_bytes = data["bytes"]
                audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_float = audio_int16.astype(np.float32) / 32768.0
                
                # Pad or truncate to exact chunk size
                if len(audio_float) < samples_per_chunk:
                    audio_float = np.pad(audio_float, (0, samples_per_chunk - len(audio_float)))
                elif len(audio_float) > samples_per_chunk:
                    audio_float = audio_float[:samples_per_chunk]
                
                # Generate
                torch.cuda.synchronize()
                t0 = time.time()
                
                audio_dq.extend(audio_float.tolist())
                audio_array = np.array(audio_dq)
                audio_embedding = get_audio_embedding(pipeline, audio_array, audio_start_idx, audio_end_idx)
                video = run_pipeline(pipeline, audio_embedding)
                
                torch.cuda.synchronize()
                gen_time = time.time() - t0
                
                # Convert frames to JPEG and send
                frames_jpeg = frames_to_jpeg_bytes(video.cpu(), quality=85)
                
                # Send metadata
                await ws.send_json({
                    "chunk_idx": chunk_idx,
                    "num_frames": len(frames_jpeg),
                    "gen_time_ms": round(gen_time * 1000, 1),
                    "fps": round(len(frames_jpeg) / gen_time, 1) if gen_time > 0 else 0,
                })
                
                # Send frames as binary (each prefixed with 4-byte length)
                for frame_jpg in frames_jpeg:
                    length_prefix = len(frame_jpg).to_bytes(4, byteorder='big')
                    await ws.send_bytes(length_prefix + frame_jpg)
                
                chunk_idx += 1
                if chunk_idx % 10 == 0:
                    logger.info(f"Chunk {chunk_idx}: {gen_time*1000:.0f}ms, {len(frames_jpeg)} frames")
    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await ws.send_json({"error": str(e)})
        except:
            pass
    finally:
        logger.info(f"Session ended after {chunk_idx} chunks")


@app.post("/api/generate")
async def generate_rest(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    seed: int = 42,
):
    """
    Non-streaming: upload image + full audio → get video file.
    For preview/testing, not real-time use.
    """
    import librosa
    import imageio
    import subprocess
    from datetime import datetime
    
    # Save uploads
    tmp_img = "/tmp/flashhead_api_img.png"
    tmp_audio = "/tmp/flashhead_api_audio.wav"
    
    with open(tmp_img, "wb") as f:
        f.write(await image.read())
    with open(tmp_audio, "wb") as f:
        f.write(await audio.read())
    
    # Prepare
    get_base_data(pipeline, cond_image_path_or_dir=tmp_img, base_seed=seed, use_face_crop=False)
    
    sample_rate = infer_params['sample_rate']
    tgt_fps = infer_params['tgt_fps']
    cached_audio_duration = infer_params['cached_audio_duration']
    frame_num = infer_params['frame_num']
    motion_frames_num = infer_params['motion_frames_num']
    slice_len = frame_num - motion_frames_num
    
    human_speech_array_all, _ = librosa.load(tmp_audio, sr=sample_rate, mono=True)
    
    cached_audio_length_sum = sample_rate * cached_audio_duration
    audio_end_idx = cached_audio_duration * tgt_fps
    audio_start_idx = audio_end_idx - frame_num
    audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)
    
    samples_per_chunk = slice_len * sample_rate // tgt_fps
    num_slices = len(human_speech_array_all) // samples_per_chunk
    slices = human_speech_array_all[:num_slices * samples_per_chunk].reshape(-1, samples_per_chunk)
    
    generated_list = []
    for chunk in slices:
        audio_dq.extend(chunk.tolist())
        audio_array = np.array(audio_dq)
        audio_embedding = get_audio_embedding(pipeline, audio_array, audio_start_idx, audio_end_idx)
        video = run_pipeline(pipeline, audio_embedding)
        generated_list.append(video.cpu())
    
    # Save video
    output_dir = os.path.join(PROJECT_ROOT, "api_outputs")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(output_dir, f"result_{timestamp}.mp4")
    temp_path = output_path.replace('.mp4', '_tmp.mp4')
    
    with imageio.get_writer(temp_path, format='mp4', mode='I', fps=tgt_fps, codec='h264', ffmpeg_params=['-bf', '0']) as writer:
        for frames in generated_list:
            frames_np = frames.numpy().astype(np.uint8)
            for i in range(frames_np.shape[0]):
                writer.append_data(frames_np[i])
    
    subprocess.run(['ffmpeg', '-i', temp_path, '-i', tmp_audio, '-c:v', 'copy', '-c:a', 'aac', '-shortest', output_path, '-y'], capture_output=True)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return FileResponse(output_path, media_type="video/mp4", filename=f"flashhead_{timestamp}.mp4")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
