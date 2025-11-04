#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, io, time, json, math, argparse, logging, shutil
from typing import Any, Dict, Optional, Tuple, List, Set
from concurrent.futures import (
    ThreadPoolExecutor, ProcessPoolExecutor, Future, wait, FIRST_COMPLETED, as_completed
)
from PIL import Image
from datasets import load_dataset
try:
    from datasets import Image as HFImage
except Exception:
    HFImage = None

# ---- TurboJPEG (optional acceleration; handle API diffs) ----
try:
    import numpy as np
    from turbojpeg import TurboJPEG, TJPF_RGB, TJSAMP_420
    try:
        from turbojpeg import TJFLAG_FASTDCT
    except Exception:
        TJFLAG_FASTDCT = 0
    _JPEG = TurboJPEG()
    HAS_TURBO = True
except Exception:
    HAS_TURBO = False
    TJFLAG_FASTDCT = 0

# ---- Hub prefetch ----
try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

LOG = logging.getLogger("anyedit_export")

# ------------------------------ utilities ------------------------------

def pure_image_id(raw_id: Optional[Any], fallback_idx: int) -> str:
    if raw_id is None:
        return f"{fallback_idx:012d}"
    s = str(raw_id).strip()
    if not s:
        return f"{fallback_idx:012d}"
    if s.isdigit():
        return f"{int(s):012d}"
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s[:128] if s else f"{fallback_idx:012d}"

def atomic_write_json(path: str, data: List[Dict[str, Any]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def get_size(path: str) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.size

def encode_jpeg_fast(rgb_img: Image.Image, out_path: str, quality: int = 90) -> Tuple[int, int]:
    """Prefer TurboJPEG (FASTDCT if available); fallback to Pillow (optimize=False)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if HAS_TURBO:
        arr = np.asarray(rgb_img, dtype=np.uint8)
        try:
            bs = _JPEG.encode(
                arr, quality=quality, pixel_format=TJPF_RGB,
                subsampling=TJSAMP_420, flags=TJFLAG_FASTDCT
            )
        except TypeError:
            # Older PyTurboJPEG uses 'jpeg_subsample'
            bs = _JPEG.encode(
                arr, quality=quality, pixel_format=TJPF_RGB,
                jpeg_subsample=TJSAMP_420, flags=TJFLAG_FASTDCT
            )
        with open(out_path, "wb") as f:
            f.write(bs)
        return rgb_img.size
    else:
        rgb_img.save(
            out_path, format="JPEG",
            quality=quality, optimize=False,
            progressive=False, subsampling=2  # NOTE: int 2
        )
        return rgb_img.size

def feature_to_pil(imgfeat: Any) -> Image.Image:
    """
    Robustly convert HF datasets Image feature (dict with 'bytes'/'path'),
    raw bytes, or PIL.Image into a PIL RGB image. No temp file is written.
    """
    # dict from datasets
    if isinstance(imgfeat, dict):
        b = imgfeat.get("bytes", None)
        if b is not None:
            im = Image.open(io.BytesIO(b))
            return im.convert("RGB") if im.mode != "RGB" else im
        p = imgfeat.get("path", None)
        if p and os.path.isabs(p) and os.path.isfile(p):
            im = Image.open(p)
            return im.convert("RGB") if im.mode != "RGB" else im
        # If it's a non-existing "filename" metadata, we'll fail below.
    # raw bytes
    if isinstance(imgfeat, (bytes, bytearray)):
        im = Image.open(io.BytesIO(imgfeat))
        return im.convert("RGB") if im.mode != "RGB" else im
    # already a PIL image or file-like
    try:
        im = Image.open(imgfeat)
        return im.convert("RGB") if im.mode != "RGB" else im
    except Exception:
        pass
    raise ValueError("Unsupported image feature payload; cannot decode in-memory")

def materialize_to_path(imgfeat: Any, spool_dir: str, stem: str) -> Tuple[str, Optional[str]]:
    """
    Fallback path: ensure a real readable file path for process backend.
    Returns (path, tmp_path_to_cleanup_or_None).
    - If absolute path exists: return it.
    - Else decode to PNG under spool_dir and return that path.
    """
    if isinstance(imgfeat, dict):
        p = imgfeat.get("path")
        if p and os.path.isabs(p) and os.path.isfile(p):
            return p, None
        b = imgfeat.get("bytes")
        if b is not None:
            tmp = os.path.join(spool_dir, f"{stem}.png")
            os.makedirs(os.path.dirname(tmp), exist_ok=True)
            with open(tmp, "wb") as f:
                f.write(b)
            return tmp, tmp
    if isinstance(imgfeat, (bytes, bytearray)):
        tmp = os.path.join(spool_dir, f"{stem}.png")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with open(tmp, "wb") as f:
            f.write(imgfeat)
        return tmp, tmp
    try:
        with Image.open(imgfeat) as im:
            tmp = os.path.join(spool_dir, f"{stem}.png")
            os.makedirs(os.path.dirname(tmp), exist_ok=True)
            im.save(tmp, format="PNG")
            return tmp, tmp
    except Exception:
        pass
    raise ValueError("Unsupported image feature payload; cannot materialize to path")

# ------------------------------ workers ------------------------------

def worker_mem_pair(src_im: Image.Image, tgt_im: Image.Image,
                    src_out: str, tgt_out: str, quality: int) -> Tuple[int, int]:
    w, h = encode_jpeg_fast(src_im, src_out, quality)
    encode_jpeg_fast(tgt_im, tgt_out, quality)
    return (w, h)

def reencode_path_to_jpeg(in_path: str, out_path: str, quality: int) -> Tuple[int, int]:
    with Image.open(in_path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        return encode_jpeg_fast(im, out_path, quality)

def worker_path_pair(src_in: str, tgt_in: str,
                     src_out: str, tgt_out: str, quality: int) -> Tuple[int, int]:
    w, h = reencode_path_to_jpeg(src_in, src_out, quality)
    reencode_path_to_jpeg(tgt_in, tgt_out, quality)
    return (w, h)

# ------------------------------ main ------------------------------

def main():
    ap = argparse.ArgumentParser(description="AnyEdit -> <image_id>_src.jpg/_tgt.jpg + incremental data_info.json (v7.3, thread+mem-encode default)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--subset-every", type=int, default=1, help="Keep 1/N by idx%N==0")
    ap.add_argument("--max-samples", type=int, default=None, help="Cap AFTER subsetting & prompt filter")

    # Defaults: thread + mem-encode
    ap.add_argument("--backend", choices=["thread","process"], default="thread",
                    help="thread (default) shares memory; process uses IPC (avoid big bytes).")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--mem-encode", dest="mem_encode", action="store_true", default=True,
                    help="In-memory decode+encode (default True; only meaningful for thread backend).")
    ap.add_argument("--no-mem-encode", dest="mem_encode", action="store_false")

    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--reencode-always", action="store_true",
                    help="Ignore any fast-copy; always re-encode (mem-encode path always re-encodes anyway).")

    ap.add_argument("--log-every", type=int, default=1000, help="Flush by finished count")
    ap.add_argument("--flush-secs", type=int, default=20, help="Flush JSON every N seconds")
    ap.add_argument("--max-inflight-mult", type=int, default=1, help="Pending <= workers * mult (default 1 for stability)")

    ap.add_argument("--streaming", action="store_true", help="Use streaming; default non-streaming")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--no-hub-prefetch", action="store_true")
    ap.add_argument("--spool-dir", default=None, help="Temp dir for process backend (suggest /dev/shm/...)")

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    os.makedirs(args.out_dir, exist_ok=True)

    # Hub prefetch (parallel, resumable-by-default)
    if not args.no_hub_prefetch and not args.streaming and snapshot_download is not None:
        LOG.info("Hub prefetch (parallel) -> ~/.cache/huggingface/hub ...")
        snapshot_download(repo_id="Bin1117/AnyEdit", repo_type="dataset")
        LOG.info("Hub prefetch done.")

    # Load dataset
    LOG.info("Loading dataset | streaming=%s", args.streaming)
    if args.streaming:
        ds = load_dataset("Bin1117/AnyEdit", split=args.split, streaming=True, cache_dir=args.cache_dir)
        total_len = None
        cast_ok = False
    else:
        ds = load_dataset("Bin1117/AnyEdit", split=args.split, cache_dir=args.cache_dir)
        total_len = len(ds)
        cast_ok = False
        if HFImage is not None:
            try:
                ds = ds.cast_column("image_file", HFImage(decode=False))
                ds = ds.cast_column("edited_file", HFImage(decode=False))
                cast_ok = True
            except Exception as e:
                LOG.warning("cast_column(decode=False) ineffective or failed: %s", e)

    # If backend=process but mem-encode requested, disable mem-encode to avoid big IPC
    if args.backend == "process" and args.mem_encode:
        LOG.warning("mem-encode is disabled for process backend to avoid large IPC. Using spool instead.")
        args.mem_encode = False

    # Prepare spool_dir if needed
    spool_dir = args.spool_dir or os.path.join(args.out_dir, ".spool_tmp")
    if not args.mem_encode or args.backend == "process":
        os.makedirs(spool_dir, exist_ok=True)

    def base_iter():
        if args.streaming:
            from itertools import count
            for i, exm in zip(count(0), ds):
                yield i, exm
        else:
            for i in range(total_len):
                yield i, ds[i]

    def keep_idx(i: int) -> bool:
        n = max(1, args.subset_every)
        return (i % n) == 0

    # Pool & state
    Exe = ThreadPoolExecutor if args.backend == "thread" else ProcessPoolExecutor
    pool = Exe(max_workers=args.workers)
    pending: Set[Future] = set()

    results: List[Dict[str, Any]] = []
    json_path = os.path.join(args.out_dir, "data_info.json")
    max_inflight = (args.workers or os.cpu_count() or 8) * max(1, args.max_inflight_mult)

    start = time.time()
    last_flush_time = start
    last_flushed_finished = 0

    submitted = 0
    finished = 0
    kept_counter = 0
    kept_total_est = None
    if (not args.streaming) and (not args.mem_encode) and (total_len is not None):
        # For ETA in mem-encode as well, we can use this rough estimate too
        kept_total_est = math.ceil(total_len / max(1, args.subset_every))
    elif (not args.streaming) and (total_len is not None):
        kept_total_est = math.ceil(total_len / max(1, args.subset_every))

    def drain_done_nonblock():
        nonlocal finished, last_flush_time, last_flushed_finished
        drained = 0
        while pending:
            done, _ = wait(pending, timeout=0, return_when=FIRST_COMPLETED)
            if not done:
                break
            for fut in done:
                try:
                    w, h = fut.result()
                except Exception as e:
                    LOG.warning("Task failed: %s", e)
                    # cleanup temps if any
                    for p in getattr(fut, "tmp_paths", []):
                        try: os.remove(p)
                        except OSError: pass
                    continue
                for p in getattr(fut, "tmp_paths", []):
                    try: os.remove(p)
                    except OSError: pass
                meta = fut.meta
                results.append({
                    "height": int(h), "width": int(w),
                    "ratio": float(w / h) if h else None,
                    "path": meta["src_name"],
                    "prompt": meta["prompt"],
                    "path_edit": meta["tgt_name"],
                })
                finished += 1
                drained += 1
            pending.difference_update(done)

        if drained > 0:
            elapsed = time.time() - start
            speed = finished / max(elapsed, 1e-9)
            if finished % args.log_every == 0:
                eta_txt = "warming up" if finished == 0 or speed == 0 else (
                    f"{max((kept_total_est or finished)-finished,0)/max(speed,1e-9)/60:.1f} min"
                )
                LOG.info("Progress %d/%s | speed %.2f it/s | ETA %s",
                         finished, str(kept_total_est) if kept_total_est else "?", speed, eta_txt)
            if (finished - last_flushed_finished >= args.log_every) or (time.time() - last_flush_time >= args.flush_secs):
                atomic_write_json(json_path, results)
                last_flushed_finished = finished
                last_flush_time = time.time()

    LOG.info("Dispatching... subset-every=%d backend=%s workers=%s mem-encode=%s TurboJPEG=%s",
             args.subset_every, args.backend, args.workers or "auto", args.mem_encode, HAS_TURBO)

    # Submission loop
    for idx, exm in base_iter():
        if not keep_idx(idx):
            if submitted and (submitted % max(1, args.log_every // 2) == 0):
                drain_done_nonblock()
            continue

        prompt = (exm.get("output") or "").strip()
        if not prompt:
            if submitted and (submitted % max(1, args.log_every // 2) == 0):
                drain_done_nonblock()
            continue

        if args.max_samples is not None and kept_counter >= args.max_samples:
            break

        stem = pure_image_id(exm.get("image_id"), idx)
        src_name = f"{stem}_src.jpg"; src_out = os.path.join(args.out_dir, src_name)
        tgt_name = f"{stem}_tgt.jpg"; tgt_out = os.path.join(args.out_dir, tgt_name)

        # Reuse
        if os.path.exists(src_out) and os.path.exists(tgt_out):
            w, h = get_size(src_out)
            results.append({"height": int(h), "width": int(w), "ratio": float(w / h) if h else None,
                            "path": src_name, "prompt": prompt, "path_edit": tgt_name})
            finished += 1
            kept_counter += 1
            if finished % args.log_every == 0 or (time.time() - last_flush_time >= args.flush_secs):
                elapsed = time.time() - start
                speed = finished / max(elapsed, 1e-9)
                eta_txt = "warming up" if finished == 0 or speed == 0 else (
                    f"{max((kept_total_est or finished)-finished,0)/max(speed,1e-9)/60:.1f} min"
                )
                LOG.info("Reused %d/%s | speed %.2f it/s | ETA %s",
                         finished, str(kept_total_est) if kept_total_est else "?", speed, eta_txt)
                atomic_write_json(json_path, results)
                last_flushed_finished = finished
                last_flush_time = time.time()
            continue

        # Prepare inputs for chosen backend/path
        if args.mem_encode and args.backend == "thread":
            # pure in-memory: decode now, pass PIL to thread
            try:
                src_im = feature_to_pil(exm["image_file"])
                tgt_im = feature_to_pil(exm["edited_file"])
            except Exception as e:
                LOG.warning("Skip idx=%d (decode in-memory failed): %s", idx, e)
                continue
            fut = pool.submit(worker_mem_pair, src_im, tgt_im, src_out, tgt_out, args.jpeg_quality)
            fut.meta = {"src_name": src_name, "tgt_name": tgt_name, "prompt": prompt}
            fut.tmp_paths = []
        else:
            # process backend or mem-encode off -> use spool/paths
            try:
                src_in, tmp1 = materialize_to_path(exm["image_file"], spool_dir, f"{stem}_src")
                tgt_in, tmp2 = materialize_to_path(exm["edited_file"], spool_dir, f"{stem}_tgt")
            except Exception as e:
                LOG.warning("Skip idx=%d (materialize failed): %s", idx, e)
                continue
            fut = pool.submit(worker_path_pair, src_in, tgt_in, src_out, tgt_out, args.jpeg_quality)
            fut.meta = {"src_name": src_name, "tgt_name": tgt_name, "prompt": prompt}
            fut.tmp_paths = [p for p in (tmp1, tmp2) if p]

        pending.add(fut)
        submitted += 1
        kept_counter += 1

        # inflight limit + intermittent drain
        while len(pending) >= max_inflight:
            drain_done_nonblock()
            time.sleep(0.01)
        if submitted % max(1, args.log_every // 2) == 0:
            drain_done_nonblock()
            elapsed = time.time() - start
            speed = (finished / max(elapsed, 1e-9))
            eta_txt = "warming up" if finished == 0 or speed == 0 else (
                f"{max((kept_total_est or finished)-finished,0)/max(speed,1e-9)/60:.1f} min"
            )
            LOG.info("Queued %d | Completed %d | Active %d | speed %.2f it/s | ETA %s",
                     submitted, finished, len(pending), speed, eta_txt)

    # Drain remaining
    LOG.info("Waiting for %d tasks...", len(pending))
    for fut in as_completed(list(pending)):
        try:
            w, h = fut.result()
        except Exception as e:
            LOG.warning("Task failed: %s", e)
            for p in getattr(fut, "tmp_paths", []):
                try: os.remove(p)
                except OSError: pass
            continue
        for p in getattr(fut, "tmp_paths", []):
            try: os.remove(p)
            except OSError: pass
        meta = fut.meta
        results.append({"height": int(h), "width": int(w), "ratio": float(w / h) if h else None,
                        "path": meta["src_name"], "prompt": meta["prompt"], "path_edit": meta["tgt_name"]})
        finished += 1
        if finished % args.log_every == 0 or (time.time() - last_flush_time >= args.flush_secs):
            elapsed = time.time() - start
            speed = finished / max(elapsed, 1e-9)
            eta_txt = "warming up" if finished == 0 or speed == 0 else (
                f"{max((kept_total_est or finished)-finished,0)/max(speed,1e-9)/60:.1f} min"
            )
            LOG.info("Progress %d/%s | speed %.2f it/s | ETA %s",
                     finished, str(kept_total_est) if kept_total_est else "?", speed, eta_txt)
            atomic_write_json(json_path, results)
            last_flushed_finished = finished
            last_flush_time = time.time()

    atomic_write_json(json_path, results)
    pool.shutdown(wait=True)

    elapsed = time.time() - start
    speed = finished / max(elapsed, 1e-9)
    LOG.info("Done. %d records written. Elapsed: %.1fs (%.2f it/s)", len(results), elapsed, speed)
    LOG.info("Output: %s | TurboJPEG: %s | mem-encode: %s", args.out_dir, HAS_TURBO, args.mem_encode)

if __name__ == "__main__":
    main()
