import argparse
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


def wait_for_history(prompt_id, timeout=1800):
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
        time.sleep(4)
    raise TimeoutError(f"Prompt {prompt_id} timed out")


def build_shot_graph(
    *,
    shot_image: str,
    prompt_text: str,
    prefix: str,
    seed: int,
    steps: int,
    use_turbo: bool,
    ref_audio: str | None = None,
    width: int = 1344,
    height: int = 768,
    length: int = 192,
):
    g = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                "weight_dtype": "default",
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
        "6": {"class_type": "LoadImage", "inputs": {"image": "damon-ref.jpg"}},
        "18": {"class_type": "LoadImage", "inputs": {"image": shot_image}},
        "8": {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "inputs": {
                "clip": ["3", 0],
                "vae": ["4", 0],
                "audio_vae": ["5", 0],
                "prompt": prompt_text,
                "width": width,
                "height": height,
                "length": length,
                "ref_image_size": "match",
                "ref_images.ref_image_0": ["6", 0],
                "ref_images.ref_image_1": ["18", 0],
            },
        },
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "12": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
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
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["4", 0]}},
        "15": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["13", 0], "vae": ["5", 0]}},
        "16": {"class_type": "CreateVideo", "inputs": {"images": ["14", 0], "audio": ["15", 0], "fps": 24}},
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

    if ref_audio:
        g["7"] = {"class_type": "LoadAudio", "inputs": {"audio": ref_audio}}
        g["8"]["inputs"]["ref_audios.ref_audio_0"] = ["7", 0]

    model_ref = ["1", 0]
    if use_turbo:
        g["2"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
                "strength_model": 1.0,
            },
        }
        model_ref = ["2", 0]

    g["9"] = {"class_type": "BasicGuider", "inputs": {"model": model_ref, "conditioning": ["8", 0]}}
    g["11"] = {
        "class_type": "BasicScheduler",
        "inputs": {
            "model": model_ref,
            "scheduler": "simple",
            "steps": steps,
            "denoise": 1.0,
        },
    }
    return g


SHOTS = [
    {
        "name": "shot1_corridor",
        "shot_image": "shot1_corridor.jpg",
        "ref_audio": None,
        "seed": 1001,
        "prompt": (
            "<Subject 1> is Lord Damon, whose face and glasses come from <Picture 1> and whose weathered wolf-fur cloak, tarnished boiled-steel breastplate, and cold stone corridor setting come from <Picture 2>.\n"
            "<Picture 2> is the exact starting frame of this shot.\n"
            "Live-action 35mm HBO prestige television cinematography, desaturated slate-blue Winterfell shadows, dim warm candlelit chiaroscuro, subtle film grain, natural motion blur, NO lens flare.\n"
            "Slow Steadicam tracking shot following <Subject 1> as he strides urgently down the damp stone corridor clutching his stomach with his gloved hand in grim physical distress, sweat glistening on his brow, breath visible in the cold air.\n"
            "Heavy leather boots echoing on wet flagstones, creaking boiled-steel armor, distant ominous orchestral cello drone."
        ),
    },
    {
        "name": "shot2_confession",
        "shot_image": "shot2_confession.jpg",
        "ref_audio": "damon_test.mp3",
        "seed": 1002,
        "prompt": (
            "<Subject 1> is Lord Damon, whose face and glasses come from <Picture 1> and whose 85mm close-up framing, wolf-fur cloak, tarnished steel armor, and dim candlelit stone background come from <Picture 2>.\n"
            "<Picture 2> is the exact starting frame of this shot.\n"
            "<Audio 1> is the exact spoken dialogue audio track for <Subject 1> (S1) and is reused verbatim, tightly lip-synced to <Subject 1>'s mouth.\n"
            "Live-action 35mm HBO prestige television cinematography, shallow depth of field, warm Rembrandt candlelight on one cheek, cold slate-blue shadow on the other, subtle film grain, NO lens flare.\n"
            "<Subject 1> (S1) looks off-camera with grave dread and whispers the exact words of <Audio 1> with subtle head tension and realistic facial micro-expressions.\n"
            "Ominous low cello drone and faint crackling candle flame."
        ),
    },
    {
        "name": "shot3_throne_reveal",
        "shot_image": "shot3_throne_reveal.jpg",
        "ref_audio": None,
        "seed": 1003,
        "prompt": (
            "<Subject 1> is Lord Damon, whose face and glasses come from <Picture 1> and whose gothic stone privy hall, flung-open heavy oak doors, and blackened-sword Iron Throne built around a gleaming white porcelain toilet come from <Picture 2>.\n"
            "<Picture 2> is the exact starting frame of this shot.\n"
            "Live-action 35mm HBO prestige television cinematography, cold blue moonlight streaming through the tall gothic lancet window, flickering floor braziers, drifting dust motes, subtle film grain, NO lens flare.\n"
            "Slow cinematic push-in as <Subject 1> staggers urgently across the flagstones toward the stone dais and climbs toward the Iron Toilet, wolf-fur cloak billowing behind him.\n"
            "Echoing boots in a vast stone hall, creaking armor, sweeping tragic orchestral strings and deep brass."
        ),
    },
    {
        "name": "shot4_cataclysm",
        "shot_image": "shot4_cataclysm.jpg",
        "ref_audio": "damon_shot4.mp3",
        "seed": 1004,
        "prompt": (
            "<Subject 1> is Lord Damon, whose face and glasses come from <Picture 1> and whose seated pose gripping the blackened iron sword-armrests of the Iron Toilet, gothic lightning window, and backlit brown liquid splatter come from <Picture 2>.\n"
            "<Picture 2> is the exact starting frame of this shot.\n"
            "<Audio 1> is the exact spoken dialogue audio track for <Subject 1> (S1) and is reused verbatim at the end of the shot, tightly lip-synced to <Subject 1>'s mouth.\n"
            "Live-action 35mm HBO prestige television cinematography, dramatic cold blue lightning flashes, sputtering wall torches in smoke, subtle film grain, NO lens flare.\n"
            "Slow-motion dramatic push-in as <Subject 1> braces on the Iron Toilet amid a chaotic backlit spray against the stone wall behind him, then stares deadpan into the camera and whispers <Audio 1>.\n"
            "Thunderous echoing low-end boom, hissing torches, tragic choir sting."
        ),
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--turbo", action="store_true")
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--length", type=int, default=192)
    args = ap.parse_args()

    for s in SHOTS:
        t0 = time.time()
        print(f"=== Starting {s['name']} (steps={args.steps}, turbo={args.turbo}, {args.width}x{args.height}) ===", flush=True)
        graph = build_shot_graph(
            shot_image=s["shot_image"],
            prompt_text=s["prompt"],
            prefix=f"video/got_{s['name']}",
            seed=s["seed"],
            steps=args.steps,
            use_turbo=args.turbo,
            ref_audio=s["ref_audio"],
            width=args.width,
            height=args.height,
            length=args.length,
        )
        pid = post_prompt(graph)["prompt_id"]
        entry = wait_for_history(pid, timeout=1800)
        dt = time.time() - t0
        print(f"=== Finished {s['name']} in {dt:.1f}s ===", flush=True)
        print(json.dumps(entry.get("outputs", {}), indent=2), flush=True)


if __name__ == "__main__":
    main()
