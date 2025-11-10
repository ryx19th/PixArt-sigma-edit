#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多进程 + 多线程版：
  1) 从已有 data_info.json (root 下) 读取所有样本，解析 path_src -> image_id
  2) 在 Bin1117/AnyEdit 里并行 filter 出这些 image_id 对应的行，建立 image_id -> 文本信息 的索引
  3) 多线程遍历旧记录，拼出带额外字段的新记录，写入 data_info_new.json

新 JSON 每条包含：
  height, width, ratio,
  path_src, path, prompt,
  prompt_src (input),
  prompt_inst (edit_instruction),
  edit_type,
  "edited object",
  "visual_input"（按你要求做 JSON-safe 清洗）
"""

import os
import re
import json
import time
import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset

try:
    from tqdm import tqdm
except Exception:  # 没装 tqdm 就用一个空壳
    def tqdm(x, **kwargs):
        return x

LOG = logging.getLogger("anyedit_meta_extra_multi")


# ---------- 小工具 ----------

def atomic_write_json(path: str, data: List[Dict[str, Any]]) -> None:
    """原子写 JSON，避免中途中断写坏。"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def extract_image_id_from_path(path_src: str) -> Optional[str]:
    """
    从 path_src 里解析 image_id：
      "COCO_train2014_000000379340_src.jpg" -> "COCO_train2014_000000379340"
    """
    base = os.path.basename(path_src)
    name, ext = os.path.splitext(base)
    if name.endswith("_src"):
        return name[:-4]
    if name.endswith("_tgt"):
        return name[:-4]
    return name or None


def load_old_info(old_json_path: str) -> List[Dict[str, Any]]:
    """加载旧 data_info.json 列表。"""
    if not os.path.exists(old_json_path):
        raise FileNotFoundError(f"旧 JSON 不存在: {old_json_path}")
    LOG.info("加载旧 meta: %s", old_json_path)
    with open(old_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    LOG.info("旧 meta 条目数: %d", len(data))
    return data


def build_needed_id_set(old_records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
    """
    从旧记录里提取所有 image_id:
      返回 (原始记录列表, image_id -> 索引列表)
    """
    id2indices: Dict[str, List[int]] = {}
    for idx, rec in enumerate(old_records):
        path_src = rec.get("path_src") or rec.get("path") or ""
        img_id = extract_image_id_from_path(path_src)
        if not img_id:
            continue
        id2indices.setdefault(img_id, []).append(idx)
    LOG.info("从旧 meta 中提取到 %d 个不同的 image_id", len(id2indices))
    return old_records, id2indices


# ---------- visual_input 清洗 ----------

def scrub_visual_input(obj: Any) -> Any:
    """
    按你的规则清洗 visual_input：
      - 如果是 str -> 原样
      - 如果是 None -> None
      - 如果 key 不存在，上层直接传 None
      - 如果是 dict/list 等容器:
          递归处理所有叶子：
            * str  -> 原样
            * None -> None
            * 其它类型（bytes、PIL.Image、int、float、bool 等）-> "non_str"
      - 其它非容器类型 -> "non_str"
    """
    # None 直接保留（最终变成 JSON null）
    if obj is None:
        return None

    # 字符串直接保留
    if isinstance(obj, str):
        return obj

    # 容器类型：dict / list / tuple / set
    if isinstance(obj, dict):
        return {k: scrub_visual_input(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [scrub_visual_input(v) for v in obj]

    # 其余类型一律视为非 str 非空，标记为 "non_str"
    return "non_str"


# ---------- 并行索引 AnyEdit ----------

def index_anyedit_by_image_id_parallel(
    needed_ids: Dict[str, List[int]],
    split: str,
    cache_dir: Optional[str],
    index_workers: int,
    batch_size: int,
) -> Dict[str, Dict[str, Any]]:
    """
    使用 datasets.filter(num_proc=...) 并行扫描 AnyEdit：
      1) remove_columns 掉图片列，避免解码大 bytes
      2) 用 batched filter + num_proc，把 image_id 在 needed_set 里的样本筛出来
      3) 对筛出来的子集做一个小循环，构建 image_id -> 文本信息 的索引
    """
    LOG.info("加载 AnyEdit 数据集: Bin1117/AnyEdit split=%s", split)
    ds = load_dataset("Bin1117/AnyEdit", split=split, cache_dir=cache_dir)

    drop_cols = []
    for c in ("image_file", "edited_file"):
        if c in ds.column_names:
            drop_cols.append(c)
    if drop_cols:
        LOG.info("移除不需要的列（避免解码图像）：%s", drop_cols)
        ds = ds.remove_columns(drop_cols)

    total = len(ds)
    LOG.info("AnyEdit split=%s 总大小: %d", split, total)

    needed_set = set(needed_ids.keys())
    LOG.info("需要匹配的 image_id 数量: %d", len(needed_set))

    # batched filter，在多进程里并行扫全表，只保留 image_id 在 needed_set 的行
    def filt_batch(batch: Dict[str, List[Any]]) -> List[bool]:
        ids = [str(x) for x in batch["image_id"]]
        return [(_id in needed_set) for _id in ids]

    LOG.info("开始并行 filter AnyEdit（num_proc=%d, batch_size=%d）...", index_workers, batch_size)
    t0 = time.time()
    ds_keep = ds.filter(
        filt_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=index_workers,
    )
    t1 = time.time()
    LOG.info(
        "filter 完成，保留了 %d 条样本，用时 %.1fs",
        len(ds_keep),
        t1 - t0,
    )

    # 对过滤后的子集构建 image_id -> 文本信息索引（很快）
    index: Dict[str, Dict[str, Any]] = {}
    for exm in tqdm(ds_keep, total=len(ds_keep), desc="building AnyEdit index"):
        img_id = str(exm.get("image_id"))

        # 清洗 visual_input
        vi_raw = exm.get("visual_input") if "visual_input" in exm else None
        vi = scrub_visual_input(vi_raw)

        index[img_id] = {
            "input": exm.get("input"),
            "edit_instruction": exm.get("edit_instruction"),
            "output": exm.get("output"),
            "edit_type": exm.get("edit_type"),
            "edited object": exm.get("edited object") if "edited object" in exm else None,
            "visual_input": vi,
        }

    LOG.info(
        "AnyEdit 索引构建完成，覆盖 image_id 数量: %d / %d",
        len(index),
        len(needed_set),
    )
    return index


# ---------- 构建新记录 ----------

def make_new_record(
    rec: Dict[str, Any],
    anyedit_index: Dict[str, Dict[str, Any]],
    root: str,
) -> Optional[Dict[str, Any]]:
    """
    单条样本的拼装逻辑：从旧 JSON + AnyEdit 索引里合并出一条新的记录。
    不再检查 prompt 是否为空；若 AnyEdit 里找不到该 image_id 或 jpg 不存在则返回 None。
    """
    path_src = rec.get("path_src") or rec.get("path") or ""
    img_id = extract_image_id_from_path(path_src)
    if not img_id:
        return None

    exm = anyedit_index.get(img_id)
    if exm is None:
        return None

    h = int(rec.get("height", 0) or 0)
    w = int(rec.get("width", 0) or 0)
    ratio = float(rec.get("ratio", (w / h) if h else 0.0) or 0.0)

    tgt_name = rec.get("path") or rec.get("path_edit") or ""
    prompt = rec.get("prompt") or ""

    # 防御性：确认 jpg 还在
    src_abs = os.path.join(root, path_src)
    tgt_abs = os.path.join(root, tgt_name)
    if not (os.path.exists(src_abs) and os.path.exists(tgt_abs)):
        return None

    new_rec = {
        "height": h,
        "width": w,
        "ratio": ratio,
        "path_src": path_src,
        "path": tgt_name,
        "prompt": prompt,
        "prompt_src": exm.get("input") or "",
        "prompt_inst": exm.get("edit_instruction") or "",
        "edit_type": exm.get("edit_type"),
        "edited object": exm.get("edited object"),
        "visual_input": exm.get("visual_input"),
    }
    return new_rec


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Build data_info_new.json with extra fields using existing jpg + AnyEdit texts (parallel index + multi-thread)."
    )
    ap.add_argument("--root", required=True, help="包含现有 jpg 和旧 data_info.json 的目录")
    ap.add_argument("--old-json", default="data_info.json", help="旧 JSON 文件名（在 --root 下）")
    ap.add_argument("--new-json", default="data_info_new.json", help="新 JSON 文件名（在 --root 下）")
    ap.add_argument("--split", default="train", help="AnyEdit split 名称，默认 train")
    ap.add_argument("--cache-dir", default=None, help="HuggingFace datasets cache_dir，可选")
    ap.add_argument("--flush-every", type=int, default=1000, help="每处理多少条样本重写一次 new json")
    ap.add_argument("--workers", type=int, default=8, help="多线程 worker 数（用于拼 JSON），默认 8")
    ap.add_argument("--index-workers", type=int, default=16, help="并行扫描 AnyEdit 的进程数 (num_proc)，默认 16")
    ap.add_argument("--index-batch-size", type=int, default=4096, help="filter 时的 batch_size，默认 4096")

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    root = args.root
    os.makedirs(root, exist_ok=True)

    old_json_path = os.path.join(root, args.old_json)
    new_json_path = os.path.join(root, args.new_json)

    # 1) 加载旧 JSON
    old_records = load_old_info(old_json_path)

    # 2) 提取 image_id 集合
    old_records, id2indices = build_needed_id_set(old_records)

    # 3) 并行扫描 AnyEdit，构建 image_id -> 文本信息 的索引（visual_input 已清洗）
    anyedit_index = index_anyedit_by_image_id_parallel(
        id2indices,
        split=args.split,
        cache_dir=args.cache_dir,
        index_workers=args.index_workers,
        batch_size=args.index_batch_size,
    )

    # 4) 开始写新 JSON：先写成 []
    atomic_write_json(new_json_path, [])
    results: List[Dict[str, Any]] = []
    last_flushed = 0
    start_time = time.time()

    total = len(old_records)
    LOG.info("开始多线程构建新 JSON，总旧样本数: %d，threads=%d", total, args.workers)

    def worker(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return make_new_record(rec, anyedit_index, root)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for new_rec in tqdm(ex.map(worker, old_records), total=total, desc="building data_info_new"):
            if new_rec is None:
                continue
            results.append(new_rec)

            if (len(results) - last_flushed) >= args.flush_every:
                atomic_write_json(new_json_path, results)
                last_flushed = len(results)
                elapsed = time.time() - start_time
                speed = len(results) / max(elapsed, 1e-9)
                LOG.info("已写入 %d 条 | speed = %.2f it/s | 文件: %s", len(results), speed, new_json_path)

    # 5) 最后一波 flush
    atomic_write_json(new_json_path, results)
    elapsed = time.time() - start_time
    speed = len(results) / max(elapsed, 1e-9)
    LOG.info("完成，共写入 %d 条。用时 %.1fs，平均速度 %.2f it/s。", len(results), elapsed, speed)
    LOG.info("输出文件: %s", new_json_path)


if __name__ == "__main__":
    main()
