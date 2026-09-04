#!/usr/bin/env python3
"""
avatar_ws_server.py

RunPod-side real-time MuseTalk avatar renderer.

NEW PROTOCOL
============

Client -> Server:

    1. JSON:
       {
           "type": "turn_audio",
           "turnId": "<uuid>",
           "sampleRate": 24000
       }

    2. One or more binary messages:
       raw PCM16LE audio

    3. JSON:
       {
           "type": "turn_audio_end",
           "turnId": "<uuid>"
       }


Server -> Client:

    One binary MP4 packet per completed AI turn:

        [0]      0x03
        [1:37]   36-byte ASCII UUID turnId
        [37:41]  uint32 LE MP4 byte length
        [41:]    MP4 bytes

    Then:

        {
            "type": "turn_video_end",
            "turnId": "<uuid>"
        }

    On error:

        {
            "type": "error",
            "turnId": "<uuid>",
            "message": "..."
        }


The generated MP4 contains:

    MuseTalk avatar video
            +
    original AI PCM audio

so the browser only needs to play a normal <video> element.

IMPORTANT
=========

This version intentionally creates ONE MP4 for the complete Gemini AI turn.

It does NOT stream individual JPEG frames.

It also limits the pending GPU queue to ONE turn.

If a newer turn arrives while another turn is already waiting in the
queue, the older queued turn is discarded.

The currently-running GPU job cannot be safely cancelled because MuseTalk
is executing inside a thread/executor. Its result may therefore finish,
but if the websocket has disconnected it will simply not be delivered.
"""

import argparse
import asyncio
import io
import json
import os
import queue
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import wave
from types import SimpleNamespace


import cv2
import numpy as np
import torch
import websockets


# ---------------------------------------------------------------------------
# MuseTalk import
# ---------------------------------------------------------------------------

# Must be run from the MuseTalk repository root.
sys.path.insert(0, os.getcwd())

import scripts.realtime_inference as rt  # noqa: E402


# ---------------------------------------------------------------------------
# Wire protocol
# ---------------------------------------------------------------------------

# NEW: complete browser-playable MP4 packet.
VIDEO_TAG = 0x03


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_models(args):
    """
    Load MuseTalk models and expose required globals to
    scripts.realtime_inference.
    """

    device = torch.device(
        f"cuda:{args.gpu_id}"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"[Models] Using device: {device}")

    # ------------------------------------------------------------
    # MuseTalk models
    # ------------------------------------------------------------

    vae, unet, pe = rt.load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device,
    )

    timesteps = torch.tensor([0], device=device)

    pe = pe.half().to(device)

    vae.vae = vae.vae.half().to(device)

    unet.model = unet.model.half().to(device)

    # ------------------------------------------------------------
    # Whisper
    # ------------------------------------------------------------

    from transformers import WhisperModel

    print("[Models] Loading Whisper...")

    audio_processor = rt.AudioProcessor(
        feature_extractor_path=args.whisper_dir
    )

    weight_dtype = unet.model.dtype

    whisper = WhisperModel.from_pretrained(
        args.whisper_dir
    )

    whisper = whisper.to(
        device=device,
        dtype=weight_dtype
    ).eval()

    whisper.requires_grad_(False)

    # ------------------------------------------------------------
    # Face parsing
    # ------------------------------------------------------------

    if args.version == "v15":
        fp = rt.FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width,
        )
    else:
        fp = rt.FaceParsing()

    # ------------------------------------------------------------
    # IMPORTANT:
    #
    # realtime_inference.py expects these as module globals.
    # ------------------------------------------------------------

    rt.args = args
    rt.device = device
    rt.vae = vae
    rt.unet = unet
    rt.pe = pe
    rt.audio_processor = audio_processor
    rt.weight_dtype = weight_dtype
    rt.whisper = whisper
    rt.timesteps = timesteps
    rt.fp = fp

    print("[Models] MuseTalk models loaded successfully.")

    return SimpleNamespace(
        device=device,
        vae=vae,
        unet=unet,
        pe=pe,
        audio_processor=audio_processor,
        weight_dtype=weight_dtype,
        whisper=whisper,
        timesteps=timesteps,
        fp=fp,
    )


# ---------------------------------------------------------------------------
# Avatar loading
# ---------------------------------------------------------------------------

def load_avatar(args):
    """
    Load already-prepared MuseTalk avatar.
    """

    avatar_path_hint = (
        f"./results/{args.version}/avatars/{args.avatar_id}"
        if args.version == "v15"
        else f"./results/avatars/{args.avatar_id}"
    )

    if not os.path.exists(avatar_path_hint):
        print(
            f"[FATAL] Avatar '{args.avatar_id}' not found "
            f"at {avatar_path_hint}."
        )

        print(
            "        Run the one-time avatar preparation "
            "pass first."
        )

        sys.exit(1)

    avatar = rt.Avatar(
        avatar_id=args.avatar_id,
        video_path=args.video_path,
        bbox_shift=(
            0
            if args.version == "v15"
            else args.bbox_shift
        ),
        batch_size=args.batch_size,
        preparation=False,
    )

    print(
        f"[Avatar] Loaded '{args.avatar_id}' | "
        f"frames={len(avatar.frame_list_cycle)}"
    )

    return avatar


# ---------------------------------------------------------------------------
# WAV helper
# ---------------------------------------------------------------------------

def pcm_to_wav_bytes(
    pcm_bytes: bytes,
    sample_rate: int
) -> bytes:

    buf = io.BytesIO()

    with wave.open(buf, "wb") as w:

        # Mono
        w.setnchannels(1)

        # PCM16
        w.setsampwidth(2)

        # Gemini/MuseTalk audio rate
        w.setframerate(sample_rate)

        w.writeframes(pcm_bytes)

    return buf.getvalue()


# ---------------------------------------------------------------------------
# FFmpeg
# ---------------------------------------------------------------------------

def check_ffmpeg():
    """
    Verify that ffmpeg exists.
    """

    ffmpeg_path = shutil.which("ffmpeg")

    if not ffmpeg_path:
        raise RuntimeError(
            "FFmpeg was not found in PATH. "
            "Install it with: apt-get update && apt-get install -y ffmpeg"
        )

    print(f"[FFmpeg] Found: {ffmpeg_path}")

    return ffmpeg_path


def mux_frames_and_audio_to_mp4(
    frame_paths,
    wav_path,
    output_path,
    fps,
):
    """
    Convert generated JPEG frames + WAV audio into one browser-playable MP4.

    Video:
        H.264

    Audio:
        AAC

    Container:
        MP4
    """

    if not frame_paths:
        raise RuntimeError(
            "No generated avatar frames available for FFmpeg."
        )

    ffmpeg_path = check_ffmpeg()

    first_frame = frame_paths[0]

    if not os.path.exists(first_frame):
        raise RuntimeError(
            f"First generated frame does not exist: {first_frame}"
        )

    # Frames are named:
    #
    #   frame_000000.jpg
    #   frame_000001.jpg
    #   frame_000002.jpg
    #
    # so FFmpeg can consume them as an image sequence.

    frame_pattern = os.path.join(
        os.path.dirname(first_frame),
        "frame_%06d.jpg"
    )

    cmd = [
        ffmpeg_path,

        "-y",

        # --------------------------------------------------------
        # Input video
        # --------------------------------------------------------

        "-framerate",
        str(fps),

        "-i",
        frame_pattern,

        # --------------------------------------------------------
        # Input audio
        # --------------------------------------------------------

        "-i",
        wav_path,

        # --------------------------------------------------------
        # Video
        # --------------------------------------------------------

        "-c:v",
        "libx264",

        # Fast encoding is important for live interview.
        "-preset",
        "ultrafast",

        # Keep latency low.
        "-tune",
        "zerolatency",

        # Browser compatibility.
        "-pix_fmt",
        "yuv420p",

        # --------------------------------------------------------
        # Audio
        # --------------------------------------------------------

        "-c:a",
        "aac",

        "-b:a",
        "128k",

        "-ar",
        "24000",

        "-ac",
        "1",

        # --------------------------------------------------------
        # Stop when the shorter stream finishes.
        # MuseTalk frame generation is based on the same audio
        # timeline, so this keeps the output bounded.
        # --------------------------------------------------------

        "-shortest",

        # Put MP4 metadata at the beginning.
        "-movflags",
        "+faststart",

        output_path,
    ]

    print(
        "[FFmpeg] Encoding MP4..."
    )

    print(
        "[FFmpeg] Command:",
        " ".join(cmd)
    )

    started = time.time()

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    elapsed = time.time() - started

    if result.returncode != 0:

        print(
            "[FFmpeg] FAILED"
        )

        print(
            result.stderr[-5000:]
        )

        raise RuntimeError(
            "FFmpeg failed to create MP4."
        )

    if not os.path.exists(output_path):

        raise RuntimeError(
            "FFmpeg returned success but MP4 does not exist."
        )

    output_size = os.path.getsize(
        output_path
    )

    if output_size <= 0:

        raise RuntimeError(
            "Generated MP4 is empty."
        )

    print(
        f"[FFmpeg] MP4 created | "
        f"size={output_size / 1024 / 1024:.2f} MB | "
        f"time={elapsed:.2f}s"
    )


# ---------------------------------------------------------------------------
# MP4 packet encoder
# ---------------------------------------------------------------------------

def encode_video_packet(
    turn_id: str,
    mp4_bytes: bytes
) -> bytes:
    """
    Encode:

        [1 byte]   tag
        [36 bytes] UUID
        [4 bytes]  uint32 MP4 length
        [N bytes]  MP4
    """

    turn_id_bytes = turn_id.encode(
        "ascii"
    )

    if len(turn_id_bytes) != 36:

        raise ValueError(
            "turnId must be a 36-character UUID, "
            f"got {len(turn_id_bytes)} bytes"
        )

    if len(mp4_bytes) > 0xFFFFFFFF:

        raise ValueError(
            "MP4 is too large for uint32 length."
        )

    header = (
        bytes([VIDEO_TAG])
        + turn_id_bytes
        + struct.pack(
            "<I",
            len(mp4_bytes)
        )
    )

    return header + mp4_bytes


# ---------------------------------------------------------------------------
# Turn processor
# ---------------------------------------------------------------------------

class TurnProcessor:
    """
    Runs one MuseTalk turn.

    Instead of producing JPEG/audio packets, this implementation:

        1. Generates all MuseTalk frames.
        2. Saves frames temporarily as JPEG files.
        3. Uses the original PCM audio.
        4. Uses FFmpeg to create ONE MP4.
        5. Places the MP4 packet into out_q.
    """

    def __init__(
        self,
        models,
        avatar,
        args
    ):

        self.m = models

        self.avatar = avatar

        self.fps = args.fps

        self.jpeg_quality = args.jpeg_quality

        self.audio_padding_left = (
            args.audio_padding_length_left
        )

        self.audio_padding_right = (
            args.audio_padding_length_right
        )

        self.tmp_dir = args.tmp_dir

    # ------------------------------------------------------------------
    # Main rendering function
    # ------------------------------------------------------------------

    def run(
        self,
        turn_id: str,
        wav_path: str,
        out_q: "queue.Queue"
    ):
        """
        Blocking function.

        Must be executed in an executor because MuseTalk is GPU-heavy.
        """

        turn_work_dir = None

        try:

            overall_start = time.time()

            print(
                f"[Turn {turn_id}] Starting render"
            )

            # ----------------------------------------------------------
            # Create temporary working directory.
            # ----------------------------------------------------------

            turn_work_dir = tempfile.mkdtemp(
                prefix=f"{turn_id}_",
                dir=self.tmp_dir,
            )

            frames_dir = os.path.join(
                turn_work_dir,
                "frames"
            )

            os.makedirs(
                frames_dir,
                exist_ok=True
            )

            output_mp4 = os.path.join(
                turn_work_dir,
                f"{turn_id}.mp4"
            )

            # ----------------------------------------------------------
            # Read original PCM audio.
            # ----------------------------------------------------------

            with wave.open(
                wav_path,
                "rb"
            ) as wf:

                n_channels = wf.getnchannels()

                sampwidth = wf.getsampwidth()

                framerate = wf.getframerate()

                raw_pcm = wf.readframes(
                    wf.getnframes()
                )

            if n_channels != 1:

                raise RuntimeError(
                    f"Expected mono audio, got {n_channels} channels."
                )

            if sampwidth != 2:

                raise RuntimeError(
                    f"Expected PCM16 audio, got "
                    f"sample width={sampwidth}."
                )

            bytes_per_sample = (
                sampwidth * n_channels
            )

            print(
                f"[Turn {turn_id}] "
                f"audio={len(raw_pcm)} bytes | "
                f"sample_rate={framerate}"
            )

            # ----------------------------------------------------------
            # Extract Whisper features.
            # ----------------------------------------------------------

            t0 = time.time()

            whisper_input_features, librosa_length = (
                self.m.audio_processor.get_audio_feature(
                    wav_path,
                    weight_dtype=self.m.weight_dtype
                )
            )

            whisper_chunks = (
                self.m.audio_processor.get_whisper_chunk(
                    whisper_input_features,
                    self.m.device,
                    self.m.weight_dtype,
                    self.m.whisper,
                    librosa_length,
                    fps=self.fps,
                    audio_padding_length_left=(
                        self.audio_padding_left
                    ),
                    audio_padding_length_right=(
                        self.audio_padding_right
                    ),
                )
            )

            video_num = len(
                whisper_chunks
            )

            print(
                f"[Turn {turn_id}] "
                f"audio feature extraction "
                f"{(time.time() - t0) * 1000:.0f}ms | "
                f"frames={video_num}"
            )

            if video_num <= 0:

                raise RuntimeError(
                    "MuseTalk generated zero video frames."
                )

            # ----------------------------------------------------------
            # Create MuseTalk data generator.
            # ----------------------------------------------------------

            gen = rt.datagen(
                whisper_chunks,
                self.avatar.input_latent_list_cycle,
                self.avatar.batch_size,
            )

            idx = 0

            n_cycle = len(
                self.avatar.coord_list_cycle
            )

            # ----------------------------------------------------------
            # Generate frames.
            # ----------------------------------------------------------

            render_start = time.time()

            with torch.no_grad():

                for whisper_batch, latent_batch in gen:

                    # --------------------------------------------------
                    # Audio embedding
                    # --------------------------------------------------

                    audio_feature_batch = self.m.pe(
                        whisper_batch.to(
                            self.m.device
                        )
                    )

                    # --------------------------------------------------
                    # Latent
                    # --------------------------------------------------

                    latent_batch = latent_batch.to(
                        device=self.m.device,
                        dtype=self.m.unet.model.dtype,
                    )

                    # --------------------------------------------------
                    # UNet
                    # --------------------------------------------------

                    pred_latents = self.m.unet.model(
                        latent_batch,
                        self.m.timesteps,
                        encoder_hidden_states=(
                            audio_feature_batch
                        ),
                    ).sample

                    pred_latents = pred_latents.to(
                        device=self.m.device,
                        dtype=self.m.vae.vae.dtype,
                    )

                    # --------------------------------------------------
                    # VAE decode
                    # --------------------------------------------------

                    recon = self.m.vae.decode_latents(
                        pred_latents
                    )

                    # --------------------------------------------------
                    # Process each generated frame
                    # --------------------------------------------------

                    for res_frame in recon:

                        bbox = (
                            self.avatar.coord_list_cycle[
                                idx % n_cycle
                            ]
                        )

                        ori_frame = (
                            self.avatar.frame_list_cycle[
                                idx % n_cycle
                            ].copy()
                        )

                        x1, y1, x2, y2 = bbox

                        try:

                            res_frame_resized = cv2.resize(
                                res_frame.astype(
                                    np.uint8
                                ),
                                (
                                    x2 - x1,
                                    y2 - y1,
                                ),
                            )

                        except Exception as resize_exc:

                            print(
                                f"[Turn {turn_id}] "
                                f"Resize failed at frame "
                                f"{idx}: {resize_exc}"
                            )

                            idx += 1

                            continue

                        # ------------------------------------------------
                        # Face mask
                        # ------------------------------------------------

                        mask = (
                            self.avatar.mask_list_cycle[
                                idx % n_cycle
                            ]
                        )

                        mask_crop_box = (
                            self.avatar.mask_coords_list_cycle[
                                idx % n_cycle
                            ]
                        )

                        # ------------------------------------------------
                        # Blend MuseTalk face with original frame
                        # ------------------------------------------------

                        combined = (
                            rt.get_image_blending(
                                ori_frame,
                                res_frame_resized,
                                bbox,
                                mask,
                                mask_crop_box,
                            )
                        )

                        # ------------------------------------------------
                        # Save JPEG.
                        #
                        # FFmpeg will later convert this sequence
                        # into H.264 MP4.
                        # ------------------------------------------------

                        frame_path = os.path.join(
                            frames_dir,
                            f"frame_{idx:06d}.jpg"
                        )

                        ok = cv2.imwrite(
                            frame_path,
                            combined,
                            [
                                cv2.IMWRITE_JPEG_QUALITY,
                                self.jpeg_quality,
                            ],
                        )

                        if not ok:

                            raise RuntimeError(
                                f"Failed to write frame "
                                f"{idx}"
                            )

                        idx += 1

            render_elapsed = (
                time.time() - render_start
            )

            print(
                f"[Turn {turn_id}] "
                f"MuseTalk rendered "
                f"{idx} frames in "
                f"{render_elapsed:.2f}s"
            )

            if idx <= 0:

                raise RuntimeError(
                    "No frames were generated."
                )

            # ----------------------------------------------------------
            # Verify frame count.
            # ----------------------------------------------------------

            frame_paths = []

            for i in range(idx):

                frame_path = os.path.join(
                    frames_dir,
                    f"frame_{i:06d}.jpg"
                )

                if os.path.exists(
                    frame_path
                ):

                    frame_paths.append(
                        frame_path
                    )

            if not frame_paths:

                raise RuntimeError(
                    "Generated frame list is empty."
                )

            # ----------------------------------------------------------
            # Create MP4.
            # ----------------------------------------------------------

            mux_start = time.time()

            mux_frames_and_audio_to_mp4(
                frame_paths=frame_paths,
                wav_path=wav_path,
                output_path=output_mp4,
                fps=self.fps,
            )

            mux_elapsed = (
                time.time() - mux_start
            )

            print(
                f"[Turn {turn_id}] "
                f"FFmpeg mux time="
                f"{mux_elapsed:.2f}s"
            )

            # ----------------------------------------------------------
            # Read complete MP4.
            # ----------------------------------------------------------

            with open(
                output_mp4,
                "rb"
            ) as f:

                mp4_bytes = f.read()

            if not mp4_bytes:

                raise RuntimeError(
                    "Generated MP4 contains no bytes."
                )

            # ----------------------------------------------------------
            # Build WebSocket packet.
            # ----------------------------------------------------------

            packet = encode_video_packet(
                turn_id=turn_id,
                mp4_bytes=mp4_bytes,
            )

            print(
                f"[Turn {turn_id}] "
                f"MP4 size="
                f"{len(mp4_bytes) / 1024 / 1024:.2f} MB | "
                f"packet="
                f"{len(packet) / 1024 / 1024:.2f} MB"
            )

            # ----------------------------------------------------------
            # Push completed MP4 to async worker.
            # ----------------------------------------------------------

            out_q.put(
                packet
            )

            # ----------------------------------------------------------
            # Signal successful completion.
            # ----------------------------------------------------------

            out_q.put(
                None
            )

            print(
                f"[Turn {turn_id}] "
                f"COMPLETE | "
                f"total="
                f"{time.time() - overall_start:.2f}s"
            )

        except Exception as exc:

            print(
                f"[Turn {turn_id}] FAILED: {exc}"
            )

            out_q.put(
                exc
            )

        finally:

            # ----------------------------------------------------------
            # Clean temporary frame/MP4 directory.
            # ----------------------------------------------------------

            if turn_work_dir:

                try:

                    shutil.rmtree(
                        turn_work_dir,
                        ignore_errors=True
                    )

                except Exception:
                    pass


# ---------------------------------------------------------------------------
# GPU queue
# ---------------------------------------------------------------------------

# IMPORTANT:
#
# Only ONE turn is allowed to wait for the GPU.
#
# Current turn:
#     running inside GPU worker
#
# Pending:
#     maximum 1
#
# New turn replaces the old pending turn.
#
# This prevents:
#
#     queue_size=10
#     queue_size=50
#     queue_size=100
#
# which happened previously.
#
turn_jobs = asyncio.Queue(
    maxsize=1
)


# ---------------------------------------------------------------------------
# Drop old queued jobs
# ---------------------------------------------------------------------------

def cleanup_queued_job(
    job
):
    """
    Delete WAV associated with a dropped queued turn.
    """

    try:

        ws, turn_id, wav_path = job

        try:

            os.remove(
                wav_path
            )

        except OSError:
            pass

        print(
            f"[Queue] Dropped stale turn "
            f"{turn_id}"
        )

    except Exception as exc:

        print(
            f"[Queue] Cleanup error: {exc}"
        )


# ---------------------------------------------------------------------------
# Latest-turn-wins enqueue
# ---------------------------------------------------------------------------

async def enqueue_latest(
    ws,
    turn_id,
    wav_path
):
    """
    Put a turn into the queue.

    If one job is already waiting, remove it and replace it with
    the newest turn.

    The currently running GPU job is NOT cancelled.
    """

    new_job = (
        ws,
        turn_id,
        wav_path,
    )

    # ----------------------------------------------------------
    # Try normal enqueue.
    # ----------------------------------------------------------

    try:

        turn_jobs.put_nowait(
            new_job
        )

        print(
            f"[Queue] Queued turn {turn_id} | "
            f"queue_size={turn_jobs.qsize()}"
        )

        return

    except asyncio.QueueFull:

        pass

    # ----------------------------------------------------------
    # Queue is full.
    #
    # Drop the old pending job.
    # ----------------------------------------------------------

    try:

        old_job = turn_jobs.get_nowait()

        cleanup_queued_job(
            old_job
        )

        turn_jobs.task_done()

    except asyncio.QueueEmpty:

        pass

    # ----------------------------------------------------------
    # Add newest job.
    # ----------------------------------------------------------

    try:

        turn_jobs.put_nowait(
            new_job
        )

        print(
            f"[Queue] Replaced stale turn with "
            f"{turn_id} | "
            f"queue_size={turn_jobs.qsize()}"
        )

    except asyncio.QueueFull:

        # Extremely unlikely race.
        #
        # Delete the new WAV instead of leaking it.

        print(
            f"[Queue] Could not enqueue turn "
            f"{turn_id}; dropping it."
        )

        cleanup_queued_job(
            new_job
        )


# ---------------------------------------------------------------------------
# GPU worker
# ---------------------------------------------------------------------------

async def gpu_worker(
    processor: TurnProcessor
):
    """
    Single GPU worker.

    One MuseTalk model instance processes one turn at a time.
    """

    loop = asyncio.get_running_loop()

    while True:

        # --------------------------------------------------------------
        # Wait for next turn.
        # --------------------------------------------------------------

        ws, turn_id, wav_path = (
            await turn_jobs.get()
        )

        print(
            f"[Worker] Starting turn {turn_id} | "
            f"remaining_queue={turn_jobs.qsize()}"
        )

        # Queue used by the blocking renderer to return its result.
        out_q = queue.Queue()

        # --------------------------------------------------------------
        # Start GPU render in executor.
        # --------------------------------------------------------------

        fut = loop.run_in_executor(
            None,
            processor.run,
            turn_id,
            wav_path,
            out_q,
        )

        try:

            while True:

                try:

                    item = out_q.get_nowait()

                except queue.Empty:

                    if fut.done():

                        try:

                            item = out_q.get_nowait()

                        except queue.Empty:

                            # Renderer ended without putting
                            # completion marker.
                            break

                    else:

                        await asyncio.sleep(
                            0.01
                        )

                        continue

                # ------------------------------------------------------
                # Successful completion marker.
                # ------------------------------------------------------

                if item is None:

                    try:

                        await ws.send(
                            json.dumps(
                                {
                                    "type":
                                        "turn_video_end",

                                    "turnId":
                                        turn_id,
                                }
                            )
                        )

                        print(
                            f"[Worker] Sent "
                            f"turn_video_end "
                            f"{turn_id}"
                        )

                    except websockets.exceptions.ConnectionClosed:

                        print(
                            f"[Worker] Client disconnected "
                            f"before turn_video_end "
                            f"{turn_id}"
                        )

                    break

                # ------------------------------------------------------
                # Renderer exception.
                # ------------------------------------------------------

                if isinstance(
                    item,
                    Exception
                ):

                    print(
                        f"[Turn {turn_id}] "
                        f"FAILED: {item}"
                    )

                    try:

                        await ws.send(
                            json.dumps(
                                {
                                    "type": "error",
                                    "turnId": turn_id,
                                    "message": str(item),
                                }
                            )
                        )

                    except websockets.exceptions.ConnectionClosed:

                        pass

                    break

                # ------------------------------------------------------
                # MP4 binary packet.
                # ------------------------------------------------------

                try:

                    await ws.send(
                        item
                    )

                    print(
                        f"[Worker] Sent MP4 "
                        f"for turn {turn_id} | "
                        f"bytes={len(item)}"
                    )

                except websockets.exceptions.ConnectionClosed:

                    print(
                        f"[Turn {turn_id}] "
                        f"client disconnected "
                        f"before MP4 delivery"
                    )

                    # Do not continue sending.
                    break

            # ----------------------------------------------------------
            # Wait for executor completion.
            # ----------------------------------------------------------

            try:

                await fut

            except Exception as exc:

                print(
                    f"[Worker] Executor error "
                    f"for {turn_id}: {exc}"
                )

        except websockets.exceptions.ConnectionClosed:

            print(
                f"[Turn {turn_id}] "
                f"client disconnected mid-stream"
            )

            # The GPU job itself may still be running.
            # We cannot safely cancel it.

            try:

                await fut

            except Exception:
                pass

        except Exception as exc:

            print(
                f"[Worker] Unexpected error "
                f"for {turn_id}: {exc}"
            )

            try:

                await fut

            except Exception:
                pass

        finally:

            # ----------------------------------------------------------
            # Remove WAV.
            # ----------------------------------------------------------

            try:

                os.remove(
                    wav_path
                )

            except OSError:

                pass

            # ----------------------------------------------------------
            # Mark queue item complete.
            # ----------------------------------------------------------

            turn_jobs.task_done()

            print(
                f"[Worker] Finished turn {turn_id} | "
                f"queue_size={turn_jobs.qsize()}"
            )


# ---------------------------------------------------------------------------
# Client connection
# ---------------------------------------------------------------------------

async def handle_connection(
    ws,
    tmp_dir
):
    """
    Handle one EC2 websocket connection.
    """

    peer = getattr(
        ws,
        "remote_address",
        None
    )

    print(
        f"[Server] Client connected: {peer}"
    )

    receiving_turn_id = None

    sample_rate = 24000

    pcm_chunks = []

    try:

        async for message in ws:

            # ----------------------------------------------------------
            # Binary = PCM audio.
            # ----------------------------------------------------------

            if isinstance(
                message,
                (bytes, bytearray)
            ):

                if receiving_turn_id is None:

                    print(
                        "[Server] Ignoring binary "
                        "frame with no active turn"
                    )

                    continue

                pcm_chunks.append(
                    bytes(message)
                )

                continue

            # ----------------------------------------------------------
            # JSON control message.
            # ----------------------------------------------------------

            try:

                msg = json.loads(
                    message
                )

            except json.JSONDecodeError:

                print(
                    "[Server] Ignoring invalid JSON"
                )

                continue

            mtype = msg.get(
                "type"
            )

            # ----------------------------------------------------------
            # Start turn.
            # ----------------------------------------------------------

            if mtype == "turn_audio":

                receiving_turn_id = msg[
                    "turnId"
                ]

                sample_rate = int(
                    msg.get(
                        "sampleRate",
                        24000
                    )
                )

                pcm_chunks = []

                print(
                    f"[Server] Receiving turn "
                    f"{receiving_turn_id} | "
                    f"sample_rate={sample_rate}"
                )

            # ----------------------------------------------------------
            # Finish turn.
            # ----------------------------------------------------------

            elif mtype == "turn_audio_end":

                turn_id = msg[
                    "turnId"
                ]

                if turn_id != receiving_turn_id:

                    print(
                        f"[Server] turn_audio_end "
                        f"for unexpected turn "
                        f"{turn_id}"
                    )

                    continue

                # ------------------------------------------------------
                # Combine PCM chunks.
                # ------------------------------------------------------

                pcm_bytes = b"".join(
                    pcm_chunks
                )

                receiving_turn_id = None

                pcm_chunks = []

                # ------------------------------------------------------
                # Empty audio protection.
                # ------------------------------------------------------

                if not pcm_bytes:

                    await ws.send(
                        json.dumps(
                            {
                                "type": "error",
                                "turnId": turn_id,
                                "message":
                                    "empty turn audio",
                            }
                        )
                    )

                    continue

                # ------------------------------------------------------
                # Validate PCM16.
                # ------------------------------------------------------

                if len(pcm_bytes) % 2 != 0:

                    print(
                        f"[Server] Warning: odd "
                        f"PCM byte count for "
                        f"{turn_id}"
                    )

                    pcm_bytes = (
                        pcm_bytes[:-1]
                    )

                # ------------------------------------------------------
                # Save WAV.
                # ------------------------------------------------------

                wav_path = os.path.join(
                    tmp_dir,
                    f"{turn_id}.wav"
                )

                wav_bytes = pcm_to_wav_bytes(
                    pcm_bytes,
                    sample_rate
                )

                with open(
                    wav_path,
                    "wb"
                ) as f:

                    f.write(
                        wav_bytes
                    )

                print(
                    f"[Server] Received complete turn "
                    f"{turn_id} | "
                    f"audio_bytes={len(pcm_bytes)}"
                )

                # ------------------------------------------------------
                # Latest-turn-wins queue.
                # ------------------------------------------------------

                await enqueue_latest(
                    ws,
                    turn_id,
                    wav_path
                )

            else:

                print(
                    f"[Server] Unknown message type: "
                    f"{mtype}"
                )

    except websockets.exceptions.ConnectionClosed as exc:

        print(
            f"[Server] Client disconnected: "
            f"{peer} | {exc}"
        )

    except Exception as exc:

        print(
            f"[Server] Connection error "
            f"from {peer}: {exc}"
        )

    finally:

        # --------------------------------------------------------------
        # If client disconnects while audio is being collected,
        # discard that partial turn.
        # --------------------------------------------------------------

        receiving_turn_id = None

        pcm_chunks = []

        print(
            f"[Server] Client disconnected: {peer}"
        )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():

    p = argparse.ArgumentParser()

    # ----------------------------------------------------------
    # Server
    # ----------------------------------------------------------

    p.add_argument(
        "--host",
        default="0.0.0.0"
    )

    p.add_argument(
        "--port",
        type=int,
        default=8765
    )

    # ----------------------------------------------------------
    # MuseTalk
    # ----------------------------------------------------------

    p.add_argument(
        "--version",
        default="v15",
        choices=["v1", "v15"]
    )

    p.add_argument(
        "--avatar_id",
        required=True
    )

    p.add_argument(
        "--video_path",
        required=True,
        help="Same avatar video used during preparation"
    )

    p.add_argument(
        "--bbox_shift",
        type=int,
        default=0
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=4
    )

    p.add_argument(
        "--fps",
        type=int,
        default=25
    )

    p.add_argument(
        "--jpeg_quality",
        type=int,
        default=80
    )

    # ----------------------------------------------------------
    # Audio padding
    # ----------------------------------------------------------

    p.add_argument(
        "--audio_padding_length_left",
        type=int,
        default=2
    )

    p.add_argument(
        "--audio_padding_length_right",
        type=int,
        default=2
    )

    # ----------------------------------------------------------
    # GPU
    # ----------------------------------------------------------

    p.add_argument(
        "--gpu_id",
        type=int,
        default=0
    )

    # ----------------------------------------------------------
    # VAE
    # ----------------------------------------------------------

    p.add_argument(
        "--vae_type",
        default="sd-vae"
    )

    # ----------------------------------------------------------
    # Model paths
    # ----------------------------------------------------------

    p.add_argument(
        "--unet_config",
        default="./models/musetalkV15/musetalk.json"
    )

    p.add_argument(
        "--unet_model_path",
        default="./models/musetalkV15/unet.pth"
    )

    p.add_argument(
        "--whisper_dir",
        default="./models/whisper"
    )

    # ----------------------------------------------------------
    # Face parsing
    # ----------------------------------------------------------

    p.add_argument(
        "--parsing_mode",
        default="jaw"
    )

    p.add_argument(
        "--left_cheek_width",
        type=int,
        default=90
    )

    p.add_argument(
        "--right_cheek_width",
        type=int,
        default=90
    )

    p.add_argument(
        "--extra_margin",
        type=int,
        default=14
    )

    # ----------------------------------------------------------
    # Temporary files
    # ----------------------------------------------------------

    p.add_argument(
        "--tmp_dir",
        default="/tmp/avatar_turns"
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():

    args = parse_args()

    # ----------------------------------------------------------
    # Create temp directory.
    # ----------------------------------------------------------

    os.makedirs(
        args.tmp_dir,
        exist_ok=True
    )

    # ----------------------------------------------------------
    # Verify FFmpeg before loading GPU models.
    # ----------------------------------------------------------

    print(
        "[Server] Checking FFmpeg..."
    )

    check_ffmpeg()

    # ----------------------------------------------------------
    # Load models.
    # ----------------------------------------------------------

    print(
        "[Server] Loading MuseTalk models..."
    )

    models = build_models(
        args
    )

    # ----------------------------------------------------------
    # Load avatar.
    # ----------------------------------------------------------

    avatar = load_avatar(
        args
    )

    # ----------------------------------------------------------
    # Create turn processor.
    # ----------------------------------------------------------

    processor = TurnProcessor(
        models,
        avatar,
        args
    )

    # ----------------------------------------------------------
    # Start single GPU worker.
    # ----------------------------------------------------------

    worker_task = asyncio.create_task(
        gpu_worker(
            processor
        )
    )

    # ----------------------------------------------------------
    # WebSocket handler.
    # ----------------------------------------------------------

    async def handler(ws):

        await handle_connection(
            ws,
            args.tmp_dir
        )

    # ----------------------------------------------------------
    # Start WebSocket server.
    # ----------------------------------------------------------

    print(
        f"[Server] Listening on "
        f"ws://{args.host}:{args.port}"
    )

    async with websockets.serve(
        handler,
        args.host,
        args.port,
        max_size=None,
    ):

        await worker_task


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    asyncio.run(
        main()
    )
