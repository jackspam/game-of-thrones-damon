import json
import os
import subprocess
import sys
import time
import urllib.request

COMFY_URL = "http://127.0.0.1:8188"


def wait_for_comfy(timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{COMFY_URL}/system_stats", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("ComfyUI did not start in time")


def get_json(path):
    with urllib.request.urlopen(f"{COMFY_URL}{path}", timeout=15) as r:
        return json.loads(r.read())


def post_prompt(prompt_graph):
    data = json.dumps({"prompt": prompt_graph}).encode()
    req = urllib.request.Request(
        f"{COMFY_URL}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print("HTTP ERROR:", e.code, e.read().decode(), file=sys.stderr)
        raise


def wait_for_history(prompt_id, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        hist = get_json(f"/history/{prompt_id}")
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                print("EXEC ERROR:", json.dumps(status, indent=2), file=sys.stderr)
                raise RuntimeError("ComfyUI execution failed")
            if entry.get("outputs"):
                return entry
        time.sleep(3)
    raise TimeoutError(f"Prompt {prompt_id} timed out")


def main():
    # Start ComfyUI if not already running
    started = False
    try:
        wait_for_comfy(timeout=3)
        print("ComfyUI already running")
    except Exception:
        print("Starting ComfyUI...")
        logf = open("/content/comfyui.log", "w")
        subprocess.Popen(
            ["python3", "main.py", "--listen", "127.0.0.1", "--port", "8188"],
            cwd="/content/ComfyUI",
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
        wait_for_comfy(timeout=120)
        started = True
        print("ComfyUI ready")

    # Inspect MiniMaxH3ReferenceToVideo schema
    info = get_json("/object_info/MiniMaxH3ReferenceToVideo")
    node_info = info.get("MiniMaxH3ReferenceToVideo", {})
    print("MiniMaxH3ReferenceToVideo inputs:", json.dumps(node_info.get("input", {}), indent=2))

    # Build API graph for 8s (length: max(5, round(8*24)) = 192 -> 192 % 17 = 5 -> 192)
    # Using 864x480 (16:9, 0.4 MP) for fast test clips first, 4-step turbo LoRA
    def build_graph(prompt_text, prefix, seed):
        return {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                    "weight_dtype": "default",
                },
            },
            "2": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": ["1", 0],
                    "lora_name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
                    "strength_model": 1.0,
                },
            },
            "3": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                    "type": "minimax",
                    "device": "default",
                },
            },
            "4": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"},
            },
            "5": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"},
            },
            "6": {
                "class_type": "LoadImage",
                "inputs": {"image": "damon-ref.jpg"},
            },
            "7": {
                "class_type": "LoadAudio",
                "inputs": {"audio": "damon_test.mp3"},
            },
            "8": {
                "class_type": "MiniMaxH3ReferenceToVideo",
                "inputs": {
                    "clip": ["3", 0],
                    "vae": ["4", 0],
                    "audio_vae": ["5", 0],
                    "prompt": prompt_text,
                    "width": 864,
                    "height": 480,
                    "length": 192,
                    "ref_image_size": "match",
                    "ref_images.ref_image_0": ["6", 0],
                    "ref_audios.ref_audio_0": ["7", 0],
                },
            },
            "9": {
                "class_type": "BasicGuider",
                "inputs": {"model": ["2", 0], "conditioning": ["8", 0]},
            },
            "10": {
                "class_type": "KSamplerSelect",
                "inputs": {"sampler_name": "res_multistep"},
            },
            "11": {
                "class_type": "BasicScheduler",
                "inputs": {
                    "model": ["2", 0],
                    "scheduler": "beta",
                    "steps": 4,
                    "denoise": 1.0,
                },
            },
            "12": {
                "class_type": "RandomNoise",
                "inputs": {"noise_seed": seed},
            },
            "13": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["12", 0],
                    "guider": ["9", 0],
                    "sampler": ["10", 0],
                    "sigmas": ["11", 0],
                    "latent_image": ["8", 1],
                },
            },
            "14": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["13", 0], "vae": ["4", 0]},
            },
            "15": {
                "class_type": "VAEDecodeAudio",
                "inputs": {"samples": ["13", 0], "vae": ["5", 0]},
            },
            "16": {
                "class_type": "CreateVideo",
                "inputs": {"images": ["14", 0], "audio": ["15", 0], "fps": 24},
            },
            "17": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["16", 0],
                    "filename_prefix": prefix,
                    "format": "auto",
                    "codec": "auto",
                },
            },
        }

    prompt_a = (
        "<Subject 1> is Lord Damon from <Picture 1>, wearing heavy dark fur-trimmed medieval leather and boiled steel armor.\n"
        "<Audio 1> is the voice-timbre reference for <Subject 1> (S1).\n"
        "Prestige fantasy drama cinematography, flickering torchlight in a cold stone castle corridor, shallow depth of field, anamorphic lens flare.\n"
        "Close-up tracking shot on <Subject 1> gripping his stomach in dread as sweat glistens on his brow. "
        "<Subject 1> (S1) whispers with grave intensity: <d>[English] Seven hells... the tavern mutton has betrayed me. Clear the keep. None shall survive what comes next. </d>\n"
        "Low ominous cello drone and distant thunder."
    )

    prompt_b = (
        "<Subject 1> is Lord Damon from <Picture 1>, wearing heavy dark fur-trimmed medieval leather and boiled steel armor.\n"
        "<Audio 1> is the exact spoken dialogue audio track for <Subject 1> (S1) and is reused verbatim, lip-synced to <Subject 1>'s mouth.\n"
        "Prestige fantasy drama cinematography, flickering torchlight in a cold stone castle corridor, shallow depth of field, anamorphic lens flare.\n"
        "Close-up tracking shot on <Subject 1> gripping his stomach in dread as sweat glistens on his brow, speaking the exact words of <Audio 1> in tight lip-sync.\n"
        "Low ominous cello drone and distant thunder."
    )

    for label, prompt, prefix, seed in [
        ("Test A (timbre ref)", prompt_a, "video/damon_test_a_timbre", 42),
        ("Test B (direct mp3)", prompt_b, "video/damon_test_b_direct", 43),
    ]:
        print(f"Submitting {label}...")
        res = post_prompt(build_graph(prompt, prefix, seed))
        pid = res["prompt_id"]
        print(f"{label} prompt_id={pid}")
        entry = wait_for_history(pid, timeout=600)
        print(f"{label} outputs:", json.dumps(entry.get("outputs", {}), indent=2))


if __name__ == "__main__":
    main()
