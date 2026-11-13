#!/usr/bin/env python3
"""Text-to-image benchmark: send the same prompts, one at a time, to an image server and time each image.

Speaks both APIs in use:
  tt     Tenstorrent tt-media-server  POST /v1/images/generations {prompt, width, height, ...} -> {images: [b64]}
  omni   vLLM-Omni                    POST /v1/images/generations {prompt, size: "WxH", ...}  -> {data: [{b64_json}]}

Writes <out>/result.json (per-request times + summary) and the images, so outputs can be checked by eye.
Usage: image_bench.py --api tt --url http://127.0.0.1:18001 --machine qb2-k8s --model flux.1-schnell --out DIR
"""
import argparse, base64, json, statistics, sys, time, urllib.request
from pathlib import Path

PROMPTS = [
    "a red fox in a snowy pine forest at dawn, photorealistic",
    "a lighthouse on a rocky coast during a storm, oil painting",
    "a bowl of ramen on a wooden table, overhead shot, food photography",
    "an astronaut riding a horse on mars, cinematic lighting",
    "a cozy reading nook with a cat asleep on a stack of books, watercolor",
    "a futuristic city skyline at night with neon signs reflected in rain puddles",
    "a close-up portrait of an elderly fisherman with a weathered face, black and white photo",
    "a hot air balloon festival over a valley at sunrise, wide angle",
    "a robot watering plants in a greenhouse, studio ghibli style",
    "a mountain lake with perfect reflections of snowy peaks, landscape photography",
    "a vintage red bicycle leaning against a yellow wall in a mediterranean village",
    "a macro photo of a dewdrop on a spider web at sunrise",
    "an isometric illustration of a tiny island with a cabin and a dock",
    "a golden retriever puppy playing in autumn leaves",
    "a steaming cup of coffee next to a laptop on a rainy windowsill",
    "a medieval castle on a cliff above the sea, fantasy concept art",
]


def request(api: str, url: str, prompt: str, w: int, h: int, steps: int, seed: int, timeout: float) -> tuple[float, bytes, dict]:
    if api == "tt":
        body = {"prompt": prompt, "width": w, "height": h, "num_inference_steps": steps, "seed": seed, "image_return_format": "PNG"}
    else:
        body = {"prompt": prompt, "size": f"{w}x{h}", "num_inference_steps": steps, "seed": seed, "response_format": "b64_json"}
    req = urllib.request.Request(url.rstrip("/") + "/v1/images/generations", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    dt = time.perf_counter() - t
    if api == "tt":
        img, server = d["images"][0], {"generation_time_s": d.get("generation_time")}
    else:
        img = d["data"][0]["b64_json"]
        gen = (d.get("metrics") or {}).get("stage_durations", {}).get("stage_0_gen_ms")
        server = {"generation_time_s": gen / 1000 if gen else None, "peak_memory_mb": (d.get("metrics") or {}).get("peak_memory_mb")}
    return dt, base64.b64decode(img), server


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api", choices=["tt", "omni"], required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--machine", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--hf-id", default="black-forest-labs/FLUX.1-schnell")
    p.add_argument("--out", required=True)
    p.add_argument("--size", default="1024x1024")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--rounds", type=int, default=2, help="passes over the prompt set")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--server", default="", help="server image/version, recorded in the result")
    a = p.parse_args()
    w, h = map(int, a.size.split("x"))
    out = Path(a.out); (out / "images").mkdir(parents=True, exist_ok=True)

    for i in range(a.warmup):
        dt, _, _ = request(a.api, a.url, PROMPTS[i % len(PROMPTS)], w, h, a.steps, 1000 + i, a.timeout)
        print(f"warmup {i}: {dt:.2f}s", flush=True)

    reqs = []
    t0 = time.perf_counter()
    for rnd in range(a.rounds):
        for i, prompt in enumerate(PROMPTS):
            seed = 42 + rnd
            dt, png, server = request(a.api, a.url, prompt, w, h, a.steps, seed, a.timeout)
            (out / "images" / f"r{rnd}-p{i:02d}.png").write_bytes(png)
            reqs.append({"round": rnd, "prompt_idx": i, "seed": seed, "latency_s": dt, **server})
            print(f"r{rnd} p{i:02d}: {dt:.2f}s (server {server.get('generation_time_s')})", flush=True)
    wall = time.perf_counter() - t0

    lat = sorted(r["latency_s"] for r in reqs)
    gen = [r["generation_time_s"] for r in reqs if r.get("generation_time_s")]
    summary = {
        "images": len(reqs), "wall_s": wall, "images_per_min": 60 * len(reqs) / wall,
        "latency_mean_s": statistics.mean(lat), "latency_p50_s": statistics.median(lat),
        "latency_p90_s": lat[int(0.9 * (len(lat) - 1))], "latency_min_s": lat[0], "latency_max_s": lat[-1],
        "server_gen_mean_s": statistics.mean(gen) if gen else None,
    }
    result = {"machine": a.machine, "model": a.model, "hf_id": a.hf_id, "api": a.api, "server": a.server,
              "workload": {"size": a.size, "steps": a.steps, "concurrency": 1, "prompts": len(PROMPTS), "rounds": a.rounds,
                           "warmup": a.warmup},
              "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "summary": summary, "requests": reqs}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
