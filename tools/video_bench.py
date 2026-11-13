#!/usr/bin/env python3
"""Text-to-video benchmark: send the same prompts, one at a time, to a video server and time each clip.

Speaks both APIs in use:
  tt     Tenstorrent tt-media-server  POST /v1/videos/generations (JSON) -> a job to poll, or the MP4 in sync mode.
         The served shape is fixed by the build (Wan2.2 on p300x2: 832x480, 81 frames, guidance 4.0 / 3.0).
  omni   vLLM-Omni                    POST /v1/videos (multipart form) -> a job to poll, then /v1/videos/<id>/content.
         Sent the same shape and guidance as the tt build.

Writes <out>/result.json (per-clip times + summary) and the clips.
"""
import argparse, json, statistics, sys, time, urllib.error, urllib.request, uuid
from pathlib import Path

PROMPTS = [
    "a red fox trotting through a snowy pine forest at dawn, photorealistic",
    "waves crashing against a lighthouse on a rocky coast during a storm",
    "an astronaut riding a horse across the surface of mars, cinematic",
    "a cat stretching and yawning on a sunny windowsill, close-up",
]
# Wan2.2's standard negative prompt (from the model card), sent to both servers so both run the same guided CFG
NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"
            "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


def _multipart(fields: dict) -> tuple[bytes, str]:
    b = uuid.uuid4().hex
    body = b"".join(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items())
    return body + f"--{b}--\r\n".encode(), f"multipart/form-data; boundary={b}"


def _open(req, timeout):
    return urllib.request.urlopen(req, timeout=timeout)


def _get_json(url: str, tries: int = 30):
    """A status poll that survives a dropped connection (a port-forward or the network blipping mid-job)."""
    for i in range(tries):
        try:
            with _open(url, 60) as r:
                return json.load(r)
        except (OSError, urllib.error.URLError) as e:
            if i == tries - 1:
                raise
            print(f"  poll error ({e}); retrying", flush=True)
            time.sleep(10)


def request(a, prompt: str, seed: int) -> tuple[float, bytes, dict]:
    base = a.url.rstrip("/")
    t = time.perf_counter()
    if a.api == "omni":
        w, h = a.size.split("x")
        body, ctype = _multipart({"prompt": prompt, "negative_prompt": NEGATIVE, "size": a.size, "width": w, "height": h,
                                  "num_frames": a.frames, "fps": a.fps, "num_inference_steps": a.steps,
                                  "guidance_scale": 4.0, "guidance_scale_2": 3.0, "boundary_ratio": 0.875, "seed": seed})
        # the async job API: /v1/videos/sync aborts any request after 600 s, and one Spark takes longer than that
        with _open(urllib.request.Request(base + "/v1/videos", data=body, headers={"Content-Type": ctype,
                                                                                   "Accept": "application/json"}), 120) as r:
            jid = json.load(r)["id"]
        while True:
            time.sleep(2)
            meta = _get_json(base + f"/v1/videos/{jid}")
            st = str(meta.get("status", "")).lower()
            if st == "completed":
                break
            if st in ("failed", "cancelled", "error"):
                raise RuntimeError(f"job {jid} {st}: {json.dumps(meta)[:500]}")
            if time.perf_counter() - t > a.timeout:
                raise TimeoutError(f"job {jid} still {st}")
        with _open(base + f"/v1/videos/{jid}/content", 600) as r:
            mp4 = r.read()
        inf = meta.get("inference_time_s") or (meta.get("metrics") or {}).get("inference_time_s")
        return time.perf_counter() - t, mp4, {"server_time_s": float(inf) if inf else None,
                                              "job": {k: v for k, v in meta.items() if k not in ("prompt",)}}
    body = json.dumps({"prompt": prompt, "negative_prompt": NEGATIVE, "num_inference_steps": a.steps, "seed": seed}).encode()
    with _open(urllib.request.Request(base + "/v1/videos/generations", data=body,
                                      headers={"Content-Type": "application/json"}), a.timeout) as r:
        if r.headers.get_content_type() == "video/mp4":
            return time.perf_counter() - t, r.read(), {}
        job = json.load(r)
    jid = job.get("id") or job.get("job_id")
    while True:
        time.sleep(2)
        meta = _get_json(base + f"/v1/videos/generations/{jid}")
        st = str(meta.get("status", "")).lower()
        if st in ("completed", "succeeded", "success", "done"):
            break
        if st in ("failed", "error", "cancelled"):
            raise RuntimeError(f"job {jid} {st}: {json.dumps(meta)[:500]}")
        if time.perf_counter() - t > a.timeout:
            raise TimeoutError(f"job {jid} still {st}")
    with _open(base + f"/v1/videos/generations/{jid}/download", 600) as r:
        mp4 = r.read()
    return time.perf_counter() - t, mp4, {"job": {k: v for k, v in meta.items() if k not in ("request_parameters",)}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api", choices=["tt", "omni"], required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--machine", required=True)
    p.add_argument("--model", default="wan2.2-t2v-a14b")
    p.add_argument("--hf-id", default="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    p.add_argument("--out", required=True)
    p.add_argument("--size", default="832x480")
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--prompts", type=int, default=len(PROMPTS))
    p.add_argument("--timeout", type=float, default=7200)
    p.add_argument("--server", default="", help="server image/version and parallel layout, recorded in the result")
    a = p.parse_args()
    out = Path(a.out); (out / "videos").mkdir(parents=True, exist_ok=True)

    for i in range(a.warmup):
        dt, _, _ = request(a, PROMPTS[-1 - i % len(PROMPTS)], 1000 + i)
        print(f"warmup {i}: {dt:.1f}s", flush=True)
    reqs = []
    t0 = time.perf_counter()
    for i, prompt in enumerate(PROMPTS[:a.prompts]):
        dt, mp4, info = request(a, prompt, 42)
        (out / "videos" / f"p{i:02d}.mp4").write_bytes(mp4)
        reqs.append({"prompt_idx": i, "seed": 42, "latency_s": dt, "bytes": len(mp4), **info})
        print(f"p{i:02d}: {dt:.1f}s {info.get('server_time_s', '')}", flush=True)
    wall = time.perf_counter() - t0
    lat = [r["latency_s"] for r in reqs]
    srv = [r["server_time_s"] for r in reqs if r.get("server_time_s")]
    summary = {"videos": len(reqs), "wall_s": wall, "videos_per_hour": 3600 * len(reqs) / wall,
               "latency_mean_s": statistics.mean(lat), "latency_min_s": min(lat), "latency_max_s": max(lat),
               "server_time_mean_s": statistics.mean(srv) if srv else None}
    result = {"machine": a.machine, "model": a.model, "hf_id": a.hf_id, "api": a.api, "kind": "video", "server": a.server,
              "workload": {"size": a.size, "frames": a.frames, "fps": a.fps, "steps": a.steps, "guidance": [4.0, 3.0],
                           "concurrency": 1, "prompts": len(reqs), "warmup": a.warmup},
              "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "summary": summary, "requests": reqs}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
