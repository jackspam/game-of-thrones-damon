import json
import sys
import time
import urllib.request

COMFY_URL = "http://127.0.0.1:8188"


def post_prompt(prompt_graph):
    data = json.dumps({"prompt": prompt_graph}).encode()
    req = urllib.request.Request(
        f"{COMFY_URL}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def wait_for_history(prompt_id, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with urllib.request.urlopen(f"{COMFY_URL}/history/{prompt_id}", timeout=15) as r:
            hist = json.loads(r.read())
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
    prompt_text = (
        "<Subject 1> is Lord Damon from <Picture 1>, wearing dark weathered medieval leather, boiled steel armor, and a heavy wolf-fur cloak.\n"
        "<Audio 1> is the exact spoken dialogue audio track for <Subject 1> (S1) and is reused verbatim, lip-synced to <Subject 1>'s mouth.\n"
        "Live-action 35mm film footage, prestige HBO medieval drama cinematography, dark moody candlelit stone castle corridor, cold blue moonlight from narrow arrow slits, rich deep shadows, natural skin texture, shallow depth of field, subtle film grain.\n"
        "Medium close-up tracking shot of <Subject 1> striding down the dim torchlit stone hallway clutching his stomach in dread as sweat glistens on his brow, speaking the exact words of <Audio 1> in tight lip-sync.\n"
        "Low ominous orchestral cello drone, crackling wall torches, and distant thunder."
    )
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", "strength_model": 1.0}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "6": {"class_type": "LoadImage", "inputs": {"image": "damon-ref.jpg"}},
        "7": {"class_type": "LoadAudio", "inputs": {"audio": "damon_test.mp3"}},
        "8": {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "inputs": {
                "clip": ["3", 0],
                "vae": ["4", 0],
                "audio_vae": ["5", 0],
                "prompt": prompt_text,
                "width": 1344,
                "height": 768,
                "length": 192,
                "ref_image_size": "match",
                "ref_images.ref_image_0": ["6", 0],
                "ref_audios.ref_audio_0": ["7", 0],
            },
        },
        "9": {"class_type": "BasicGuider", "inputs": {"model": ["2", 0], "conditioning": ["8", 0]}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "11": {"class_type": "BasicScheduler", "inputs": {"model": ["2", 0], "scheduler": "simple", "steps": 4, "denoise": 1.0}},
        "12": {"class_type": "RandomNoise", "inputs": {"noise_seed": 101}},
        "13": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["12", 0], "guider": ["9", 0], "sampler": ["10", 0], "sigmas": ["11", 0], "latent_image": ["8", 1]}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["4", 0]}},
        "15": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["13", 0], "vae": ["5", 0]}},
        "16": {"class_type": "CreateVideo", "inputs": {"images": ["14", 0], "audio": ["15", 0], "fps": 24}},
        "17": {"class_type": "SaveVideo", "inputs": {"video": ["16", 0], "filename_prefix": "video/damon_test_c_768p_simple", "format": "auto", "codec": "auto"}},
    }
    t0 = time.time()
    pid = post_prompt(graph)["prompt_id"]
    print(f"Test C 768p simple prompt_id={pid}", flush=True)
    entry = wait_for_history(pid, timeout=600)
    print(f"Wall time: {time.time() - t0:.1f}s", flush=True)
    print(json.dumps(entry.get("outputs", {}), indent=2))


if __name__ == "__main__":
    main()
