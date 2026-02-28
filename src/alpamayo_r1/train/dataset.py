import math
import time
import threading
import torch
import json
import os
import random
import logging
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ClipPrefetcher:
    """Asynchronous clip loader with LRU cache.

    A background thread pool loads clips ahead of time so that I/O overlaps
    with GPU computation.  An LRU cache avoids reloading clips that are
    accessed repeatedly (common with clip-grouped ordering).

    Lazy-initialises threading primitives so the object survives pickling
    by DataLoader workers (fork).
    """

    def __init__(self, max_cache_size: int = 4, num_threads: int = 2):
        self._max_size = max_cache_size
        self._num_threads = num_threads
        # Lazy-init after fork
        self._cache: OrderedDict | None = None
        self._lock: threading.Lock | None = None
        self._futures: dict | None = None
        self._executor: ThreadPoolExecutor | None = None

    def _ensure_init(self):
        if self._cache is None:
            self._cache = OrderedDict()
            self._lock = threading.Lock()
            self._futures = {}
            self._executor = ThreadPoolExecutor(max_workers=self._num_threads)

    @staticmethod
    def _load_from_disk(path: str):
        last_err = None
        for attempt in range(3):
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except OSError as e:
                last_err = e
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
        raise last_err

    def prefetch(self, clip_id: str, path: str):
        """Schedule background loading (no-op if already cached or loading)."""
        self._ensure_init()
        with self._lock:
            if clip_id in self._cache or clip_id in self._futures:
                return
            self._futures[clip_id] = self._executor.submit(
                self._load_from_disk, path
            )

    def get(self, clip_id: str, path: str):
        """Return clip data, blocking until available."""
        self._ensure_init()
        with self._lock:
            if clip_id in self._cache:
                self._cache.move_to_end(clip_id)
                return self._cache[clip_id]
            future = self._futures.pop(clip_id, None)

        data = future.result() if future is not None else self._load_from_disk(path)

        with self._lock:
            self._cache[clip_id] = data
            self._cache.move_to_end(clip_id)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
        return data

    def __getstate__(self):
        return {"_max_size": self._max_size, "_num_threads": self._num_threads}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache = None
        self._lock = None
        self._futures = None
        self._executor = None


_NO_BATCH_DIM_KEYS = {"pixel_values", "image_grid_thw"}


def collate_fn(batch):
    """Collate for batch_size=1. Adds a batch dim to each tensor in each window.

    pixel_values and image_grid_thw are excluded because their first dimension
    is num_patches / num_images, not batch.
    """
    windows = batch[0]
    return [
        {
            k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and k not in _NO_BATCH_DIM_KEYS else v
            for k, v in w.items()
        }
        for w in windows
    ]


eval_collate_fn = collate_fn


def batched_collate_fn(batch):
    """Collate for batch_size >= 1.  Stacks tensors across samples at each
    window position.

    Input : list of N items, each a list of W window dicts.
    Output: list of W window dicts with tensors of shape [N, ...].

    Keys in _NO_BATCH_DIM_KEYS (pixel_values, image_grid_thw) are concatenated
    rather than stacked, because their first dimension is num_patches / num_images
    rather than batch.
    """
    num_windows = len(batch[0])
    result = []
    for w_idx in range(num_windows):
        window: dict = {}
        for key in batch[0][w_idx]:
            values = [sample[w_idx][key] for sample in batch]
            if isinstance(values[0], torch.Tensor):
                if key in _NO_BATCH_DIM_KEYS:
                    window[key] = torch.cat(values, dim=0)
                else:
                    window[key] = torch.stack(values)
            else:
                window[key] = values[0]
        result.append(window)
    return result


class StreamingDataset(torch.utils.data.Dataset):
    """
    Dataset that creates contiguous sliding window subsequences from pre-dumped clips.

    Each item is a list of (rollout_steps + 1) consecutive sliding window inputs:
    - First `rollout_steps` windows: used for rollout (no loss computation)
    - Last window: used for loss computation (with best_cot as label)

    rollout_steps is randomly sampled from [1, max_rollout_steps] per item to
    expose the model to varying amounts of streaming error accumulation.

    N such contiguous subsequences are sampled from each clip, deduplicated by
    (start, rollout_steps).

    For streaming (non-prefill) windows, only the new frame per view is kept:
    - input_ids / attention_mask: old-frame vision tokens removed
    - pixel_values: only the last frame's patches per view
    - image_grid_thw: only the last frame's grid entry per view

    output_token_ids is stored separately per window and concatenated here so
    that output_ids_range can be easily determined.

    Expected directory layout:
        data_dir/<clip_id>/sliding_window_inputs.pt
            Each .pt contains a list of window dicts with output_token_ids stored separately.
    """

    def __init__(self, args, processor=None, training_stage="vlm"):
        self.args = args
        self.training_stage = training_stage
        self.min_rollout_steps = getattr(args, "min_rollout_steps", 1)
        self.max_rollout_steps = args.max_rollout_steps
        self.samples_per_clip = args.samples_per_clip
        self.data_dir = args.data_dir
        self.num_views = 4
        self.num_frames_per_view = 4
        self.num_pixels_per_frame = 720

        # Resolve vision token IDs for streaming extraction
        if processor is not None:
            tokenizer = processor.tokenizer
        else:
            tokenizer = getattr(args, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(
                "StreamingDataset requires a processor (pass processor=) or "
                "args.tokenizer to resolve vision token IDs"
            )
        self._vision_start_id = tokenizer.encode("<|vision_start|>")[0]
        self._vision_end_id = tokenizer.encode("<|vision_end|>")[0]
        self._endoftext_id = tokenizer.encode("<|endoftext|>")[0]
        self._traj_future_start_id = tokenizer.encode("<|traj_future_start|>")[0]

        with open(args.clip_list, "r") as f:
            self.clip_ids = json.load(f)

        self._rank = int(os.environ.get("RANK", 0))
        self._world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Prefetcher: background threads load the next clip while GPU is busy
        prefetch_cache_size = getattr(args, "prefetch_cache_size", 4)
        prefetch_threads = getattr(args, "prefetch_threads", 2)
        self._clip_prefetcher = ClipPrefetcher(
            max_cache_size=prefetch_cache_size,
            num_threads=prefetch_threads,
        )

        # Current rollout steps — set externally via set_rollout_steps() before
        # each training step so all samples in a batch share the same value.
        self._rollout_steps = self.max_rollout_steps

        self.load_data()
        self._organize_by_clip()

    # Default number of sliding windows per clip (avoids loading data at index time)
    DEFAULT_CLIP_LENGTH = 120

    def _get_clip_path(self, clip_id):
        return os.path.join(self.data_dir, clip_id, "sliding_window_inputs.pt")

    def _load_clip(self, clip_id):
        """Load clip via the prefetch cache (background I/O + LRU)."""
        return self._clip_prefetcher.get(clip_id, self._get_clip_path(clip_id))

    def load_data(self):
        """Build a flat index of (clip_id, sample_idx) pairs.

        The start position within the clip is sampled randomly in __getitem__
        using the current _rollout_steps value.
        """
        self.data: list[tuple[str, int]] = []  # (clip_id, sample_idx)
        for clip_id in self.clip_ids:
            if not os.path.isfile(self._get_clip_path(clip_id)):
                continue
            for i in range(self.samples_per_clip):
                self.data.append((clip_id, i))

    def set_rollout_steps(self, rollout_steps: int):
        """Set the rollout steps for the next batch (called by the trainer)."""
        self._rollout_steps = rollout_steps

    def _organize_by_clip(self):
        """Group samples by clip_id and build clip-grouped index ordering.

        Within each clip, samples are sorted by sample_idx.
        Clip order is shuffled (deterministic per epoch).
        For distributed training, clips are split across ranks.
        """
        groups = defaultdict(list)
        for i, entry in enumerate(self.data):
            groups[entry[0]].append(i)
        for clip_id in groups:
            groups[clip_id].sort(key=lambda i: self.data[i][1])
        self._clip_groups = groups  # clip_id -> [indices into self.data]
        self._clip_ids = list(groups.keys())
        self._epoch = 0
        self._rebuild_order()

    def _rebuild_order(self):
        """Rebuild ordered index list based on current epoch and rank."""
        clip_ids = list(self._clip_ids)
        random.Random(self._epoch).shuffle(clip_ids)

        # Distribute clips across ranks (round-robin)
        if self._world_size > 1:
            clip_ids = clip_ids[self._rank::self._world_size]

        self._ordered_indices = []
        for clip_id in clip_ids:
            self._ordered_indices.extend(self._clip_groups[clip_id])

        # Pad for DDP so each rank has the same number of samples
        if self._world_size > 1 and self._ordered_indices:
            target = math.ceil(len(self.data) / self._world_size)
            while len(self._ordered_indices) < target:
                wrap_idx = len(self._ordered_indices) % max(1, len(self._ordered_indices))
                self._ordered_indices.append(self._ordered_indices[wrap_idx])
            self._ordered_indices = self._ordered_indices[:target]

        # Precompute next-clip mapping for prefetching
        ordered_clips: list[str] = []
        prev = None
        for real_idx in self._ordered_indices:
            cid = self.data[real_idx][0]
            if cid != prev:
                ordered_clips.append(cid)
                prev = cid
        self._next_clip_for: dict[str, str] = {}
        for i in range(len(ordered_clips) - 1):
            self._next_clip_for[ordered_clips[i]] = ordered_clips[i + 1]

    def set_epoch(self, epoch):
        """Reshuffle clip order for a new epoch."""
        self._epoch = epoch
        self._rebuild_order()

    def _process_streaming_inputs(self, input_ids, attention_mask,
                                  pixel_values, image_grid_thw):
        """For streaming (non-prefill) windows, keep only new-frame tokens and pixels.

        For each of the 4 views, removes the first 3 (old) frames' vision tokens
        from input_ids / attention_mask and extracts only the last frame's
        pixel patches and grid entry.

        All inputs have a leading batch dim of 1 (as stored in the dumped data).
        """
        ids = input_ids[0]  # [seq_len]
        vision_starts = (ids == self._vision_start_id).nonzero(as_tuple=True)[0]
        vision_ends = (ids == self._vision_end_id).nonzero(as_tuple=True)[0]

        # Build keep mask: start with True everywhere, then zero-out old frames and system tokens
        keep_mask = torch.ones(input_ids.shape[1], dtype=torch.bool)
        first_vision_start = vision_starts[0].item()
        keep_mask[:first_vision_start] = False
        for view_idx in range(self.num_views):
            for frame_offset in range(self.num_frames_per_view - 1):  # first 3 frames
                gidx = view_idx * self.num_frames_per_view + frame_offset
                start = vision_starts[gidx].item()
                end = vision_ends[gidx].item()
                keep_mask[start : end + 1] = False  # vision_start through vision_end inclusive

        input_ids = input_ids[:, keep_mask]
        attention_mask = attention_mask[:, keep_mask]

        # Extract last frame's pixel patches per view
        new_pixel_values = []
        for view_idx in range(self.num_views):
            view_end = (view_idx + 1) * self.num_frames_per_view * self.num_pixels_per_frame
            view_start = view_end - self.num_pixels_per_frame
            new_pixel_values.append(pixel_values[view_start:view_end])
        pixel_values = torch.cat(new_pixel_values, dim=0)

        # Extract last frame's grid entry per view (indices 3, 7, 11, 15)
        new_grid_indices = [
            view_idx * self.num_frames_per_view + (self.num_frames_per_view - 1)
            for view_idx in range(self.num_views)
        ]
        image_grid_thw = image_grid_thw[new_grid_indices]

        return input_ids, attention_mask, pixel_values, image_grid_thw

    def __len__(self):
        return len(self._ordered_indices)

    def _build_window_item(self, w, is_streaming, with_output=False):
        """Build a single window item.

        Args:
            w: Raw window dict from dumped data.
            is_streaming: True for non-prefill windows (old-frame tokens removed).
            with_output: If True, concatenate output_token_ids to input and build labels.
                         Only the last window (loss computation) needs this.
        """
        td = w["tokenized_data"]

        if is_streaming:
            input_ids, attention_mask, pixel_values, image_grid_thw = (
                self._process_streaming_inputs(
                    td["input_ids"], td["attention_mask"],
                    td["pixel_values"], td["image_grid_thw"],
                )
            )
        else:
            input_ids = td["input_ids"]
            attention_mask = td["attention_mask"]
            pixel_values = td["pixel_values"]
            image_grid_thw = td["image_grid_thw"]

        input_ids = input_ids.squeeze(0)          # [input_len]
        attention_mask = attention_mask.squeeze(0)  # [input_len]

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "ego_history_xyz": w["ego_history_xyz"].squeeze(0),
            "ego_history_rot": w["ego_history_rot"].squeeze(0),
            "is_prefill": not is_streaming,
        }

        if with_output:
            input_len = input_ids.shape[0]
            output_token_ids = w.get("output_token_ids", None)
            if output_token_ids is not None and output_token_ids.numel() > 0:
                # Ensure output ends with <|traj_future_start|>
                if output_token_ids[-1].item() == self._endoftext_id:
                    output_token_ids = output_token_ids.clone()
                    output_token_ids[-1] = self._traj_future_start_id
                elif output_token_ids[-1].item() != self._traj_future_start_id:
                    output_token_ids = torch.cat([
                        output_token_ids,
                        torch.tensor([self._traj_future_start_id], dtype=output_token_ids.dtype),
                    ])
                output_len = output_token_ids.shape[0]
                item["input_ids"] = torch.cat([input_ids, output_token_ids])
                item["attention_mask"] = torch.cat([
                    attention_mask,
                    torch.ones(output_len, dtype=attention_mask.dtype),
                ])
                item["labels"] = torch.cat([
                    torch.full((input_len,), -100, dtype=torch.long),
                    output_token_ids,
                ])
                item["output_ids_range"] = (input_len, input_len + output_len)
            else:
                item["labels"] = torch.full((input_len,), -100, dtype=torch.long)
                item["output_ids_range"] = (input_len, input_len)

        return item

    def __getitem__(self, idx, _skip_attempts=10, _bad_clips=None):
        try:
            return self._getitem_impl(idx)
        except (OSError, Exception) as e:
            if _skip_attempts <= 0:
                raise
            real_idx = self._ordered_indices[idx]
            clip_id = self.data[real_idx][0]
            logger.warning(
                "Skip bad sample idx=%s clip_id=%s (%s), retrying with another sample: %s",
                idx, clip_id, type(e).__name__, e,
            )
            if _bad_clips is None:
                _bad_clips = set()
            _bad_clips.add(clip_id)
            others = [
                i for i in range(len(self))
                if self.data[self._ordered_indices[i]][0] not in _bad_clips
            ]
            if not others:
                raise
            return self.__getitem__(random.choice(others), _skip_attempts=_skip_attempts - 1, _bad_clips=_bad_clips)

    def _getitem_impl(self, idx):
        real_idx = self._ordered_indices[idx]
        clip_id = self.data[real_idx][0]

        # Trigger prefetch of the next clip
        next_clip = self._next_clip_for.get(clip_id)
        if next_clip is not None:
            self._clip_prefetcher.prefetch(next_clip, self._get_clip_path(next_clip))

        all_windows = self._load_clip(clip_id)
        actual_len = len(all_windows)

        seq_len = self._rollout_steps + 1
        max_start = max(0, actual_len - seq_len)
        start = random.randint(0, max_start)
        windows = all_windows[start : start + seq_len]

        last = len(windows) - 1
        items = []
        for i, w in enumerate(windows):
            if self.training_stage == "vlm":
                item = self._build_window_item(w, is_streaming=(i > 0), with_output=(i == last))
            else:  # expert
                item = self._build_window_item(w, is_streaming=(i > 0), with_output=False)
                if i == last:
                    item["ego_future_xyz"] = w["ego_future_xyz"].squeeze(0)
                    item["ego_future_rot"] = w["ego_future_rot"].squeeze(0)
            items.append(item)
        return items


class EvalStreamingDataset(StreamingDataset):
    """Eval dataset that returns entire clips for full-clip evaluation.

    Each item is a complete clip (all windows from start to end).
    - Window 0: prefill (no metrics)
    - Windows 1+: streaming windows with trajectory ground truth for minADE

    Eval runs on rank 0 only, so we skip distributed sharding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override: eval is rank-0 only, no need to shard across ranks
        self._rank = 0
        self._world_size = 1
        self._rebuild_order()

    def load_data(self):
        """One entry per clip — eval runs the entire clip."""
        self.data: list[tuple[str, int, int]] = []
        for clip_id in self.clip_ids:
            if not os.path.isfile(self._get_clip_path(clip_id)):
                continue
            self.data.append((clip_id, 0, 0))  # start=0, seq_len=0 (ignored, load full clip)

    def __getitem__(self, idx):
        real_idx = self._ordered_indices[idx]
        clip_id = self.data[real_idx][0]
        all_windows = self._load_clip(clip_id)

        result = []
        for i, w in enumerate(all_windows):
            item = self._build_window_item(w, is_streaming=(i > 0), with_output=False)
            item["ego_future_xyz"] = w["ego_future_xyz"].squeeze(0)
            item["ego_future_rot"] = w["ego_future_rot"].squeeze(0)
            item["clip_id"] = clip_id
            result.append(item)
        return result


if __name__ == "__main__":
    """Verify dataset behavior using a single clip of dumped data.

    Usage:
        python -m alpamayo_r1.train.dataset <data_dir> <clip_id>

    Example:
        python -m alpamayo_r1.train.dataset \
            /mnt/moosefs-1/users/zekail/dumped_inputs \
            00040136-e651-4abd-991d-0655ccda9430
    """
    import sys
    import tempfile
    from omegaconf import OmegaConf
    from transformers import AutoProcessor

    if len(sys.argv) < 3:
        print("Usage: python -m alpamayo_r1.train.dataset <data_dir> <clip_id>")
        sys.exit(1)

    data_dir = sys.argv[1]
    clip_id = sys.argv[2]

    NUM_VIEWS = 4
    NUM_FRAMES = 4
    PIXELS_PER_FRAME = 720
    MIN_ROLLOUT_STEPS = 1
    MAX_ROLLOUT_STEPS = 3

    # ---- Setup ----
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    tokenizer = processor.tokenizer
    vision_start_id = tokenizer.encode("<|vision_start|>")[0]
    vision_end_id = tokenizer.encode("<|vision_end|>")[0]
    image_pad_id = tokenizer.encode("<|image_pad|>")[0]
    merge_size = processor.image_processor.merge_size
    print(f"vision_start_id={vision_start_id}, vision_end_id={vision_end_id}, "
          f"image_pad_id={image_pad_id}, merge_size={merge_size}")

    clip_list_path = os.path.join(tempfile.mkdtemp(), "clip_list.json")
    with open(clip_list_path, "w") as f:
        json.dump([clip_id], f)

    # Use min_rollout_steps=2 for detailed checks so there's always a pure
    # streaming rollout window (window 1) separate from the loss window.
    cfg = OmegaConf.create({
        "data_dir": data_dir,
        "clip_list": clip_list_path,
        "min_rollout_steps": 2,
        "max_rollout_steps": MAX_ROLLOUT_STEPS,
        "samples_per_clip": 1,
    })

    # Load raw data for comparison
    raw_path = os.path.join(data_dir, clip_id, "sliding_window_inputs.pt")
    raw_windows = torch.load(raw_path, map_location="cpu", weights_only=False)
    print(f"Raw data: {len(raw_windows)} windows from {raw_path}")

    dataset = StreamingDataset(cfg, processor=processor, training_stage="vlm")
    print(f"Dataset length: {len(dataset)} (expected 1)")
    assert len(dataset) == 1

    item = dataset[0]
    # Reconstruct raw subsequence from the index for verification
    _clip_id, _start, _seq_len = dataset.data[0]
    _all_windows = torch.load(os.path.join(data_dir, _clip_id, "sliding_window_inputs.pt"),
                              map_location="cpu", weights_only=False)
    raw_subseq = _all_windows[_start : _start + _seq_len]
    num_windows = len(item)
    rollout_steps = num_windows - 1
    assert 3 <= num_windows <= MAX_ROLLOUT_STEPS + 1, \
        f"item has {num_windows} windows, expected 3..{MAX_ROLLOUT_STEPS + 1}"
    print(f"Item has {num_windows} windows (rollout_steps={rollout_steps})")

    errors = []

    def check(cond, msg):
        if not cond:
            errors.append(msg)
            print(f"  FAIL: {msg}")
        else:
            print(f"  OK:   {msg}")

    def _compute_streaming_keep_mask(raw_ids):
        """Reproduce _process_streaming_inputs keep_mask for verification."""
        vs = (raw_ids == vision_start_id).nonzero(as_tuple=True)[0]
        ve = (raw_ids == vision_end_id).nonzero(as_tuple=True)[0]
        mask = torch.ones(raw_ids.shape[0], dtype=torch.bool)
        mask[:vs[0].item()] = False
        for v in range(NUM_VIEWS):
            for f in range(NUM_FRAMES - 1):
                gidx = v * NUM_FRAMES + f
                mask[vs[gidx].item():ve[gidx].item() + 1] = False
        return mask

    def _compute_expected_pixel_values(raw_pv):
        """Extract last frame pixel patches per view."""
        parts = []
        for v in range(NUM_VIEWS):
            start = (v * NUM_FRAMES + NUM_FRAMES - 1) * PIXELS_PER_FRAME
            parts.append(raw_pv[start:start + PIXELS_PER_FRAME])
        return torch.cat(parts, dim=0)

    # ==================================================================
    # 1. Prefill window (window 0)
    # ==================================================================
    print("\n" + "=" * 60)
    print("1. PREFILL WINDOW (window 0) — pass-through from raw")
    print("=" * 60)

    w0 = item[0]
    td0 = raw_subseq[0]["tokenized_data"]

    check(w0["is_prefill"] is True, "is_prefill is True")
    check(torch.equal(w0["input_ids"], td0["input_ids"].squeeze(0)),
          "input_ids matches raw (squeezed)")
    check(torch.equal(w0["attention_mask"], td0["attention_mask"].squeeze(0)),
          "attention_mask matches raw (squeezed)")
    check(torch.equal(w0["pixel_values"], td0["pixel_values"]),
          "pixel_values matches raw")
    check(torch.equal(w0["image_grid_thw"], td0["image_grid_thw"]),
          "image_grid_thw matches raw")
    check(torch.equal(w0["ego_history_xyz"], raw_subseq[0]["ego_history_xyz"].squeeze(0)),
          "ego_history_xyz matches raw (squeezed)")
    check(torch.equal(w0["ego_history_rot"], raw_subseq[0]["ego_history_rot"].squeeze(0)),
          "ego_history_rot matches raw (squeezed)")
    check("labels" not in w0, "no labels on rollout window")
    check("output_ids_range" not in w0, "no output_ids_range on rollout window")

    # ==================================================================
    # 2. Streaming window (window 1) — extraction logic
    # ==================================================================
    print("\n" + "=" * 60)
    print("2. STREAMING WINDOW (window 1) — extraction logic")
    print("=" * 60)

    w1 = item[1]
    td1 = raw_subseq[1]["tokenized_data"]
    raw_ids1 = td1["input_ids"].squeeze(0)

    check(w1["is_prefill"] is False, "is_prefill is False")

    # 2a. Pixel values
    expected_pv = _compute_expected_pixel_values(td1["pixel_values"])
    check(w1["pixel_values"].shape == expected_pv.shape,
          f"pixel_values shape {tuple(w1['pixel_values'].shape)} == "
          f"expected {tuple(expected_pv.shape)}")
    check(torch.equal(w1["pixel_values"], expected_pv),
          "pixel_values content matches last frame per view")

    # 2b. image_grid_thw
    expected_grid_idx = [v * NUM_FRAMES + NUM_FRAMES - 1 for v in range(NUM_VIEWS)]
    check(torch.equal(w1["image_grid_thw"], td1["image_grid_thw"][expected_grid_idx]),
          f"image_grid_thw matches raw indices {expected_grid_idx}")

    # 2c. input_ids via keep_mask
    keep_mask = _compute_streaming_keep_mask(raw_ids1)
    check(torch.equal(w1["input_ids"], raw_ids1[keep_mask]),
          "input_ids matches manually computed keep_mask")

    # 2d. Vision token counts
    vs_count = (w1["input_ids"] == vision_start_id).sum().item()
    ve_count = (w1["input_ids"] == vision_end_id).sum().item()
    check(vs_count == NUM_VIEWS, f"vision_start count {vs_count} == {NUM_VIEWS}")
    check(ve_count == NUM_VIEWS, f"vision_end count {ve_count} == {NUM_VIEWS}")

    # 2e. image_pad count (accounts for spatial merge)
    grid_entry = td1["image_grid_thw"][0]
    tokens_per_image = (grid_entry[1] * grid_entry[2]).item() // (merge_size ** 2)
    expected_pad = NUM_VIEWS * tokens_per_image
    actual_pad = (w1["input_ids"] == image_pad_id).sum().item()
    check(actual_pad == expected_pad,
          f"image_pad count {actual_pad} == {expected_pad} "
          f"({NUM_VIEWS} views * {tokens_per_image} tokens/image, "
          f"merge={merge_size})")

    # 2f. attention_mask
    check(torch.equal(w1["attention_mask"], td1["attention_mask"].squeeze(0)[keep_mask]),
          "attention_mask matches keep_mask applied to raw")

    # 2g. Trajectory data
    check(torch.equal(w1["ego_history_xyz"], raw_subseq[1]["ego_history_xyz"].squeeze(0)),
          "ego_history_xyz matches raw (squeezed)")

    # ==================================================================
    # 3. Last window (loss) — output_token_ids + labels
    # ==================================================================
    print("\n" + "=" * 60)
    print("3. LAST WINDOW — output_token_ids concatenation + labels")
    print("=" * 60)

    wlast = item[-1]
    raw_last = raw_subseq[-1]
    td_last = raw_last["tokenized_data"]
    raw_ids_last = td_last["input_ids"].squeeze(0)
    raw_output_ids = raw_last["output_token_ids"]

    keep_last = _compute_streaming_keep_mask(raw_ids_last)
    streaming_ids = raw_ids_last[keep_last]
    input_len = streaming_ids.shape[0]

    # input_ids = streaming_input + output_token_ids
    expected_full_ids = torch.cat([streaming_ids, raw_output_ids])
    check(torch.equal(wlast["input_ids"], expected_full_ids),
          "input_ids = streaming_input ++ output_token_ids")

    # labels
    expected_labels = torch.cat([
        torch.full((input_len,), -100, dtype=torch.long),
        raw_output_ids,
    ])
    check(torch.equal(wlast["labels"], expected_labels),
          "labels: -100 for input portion, output_token_ids for output portion")

    # output_ids_range
    start, end = wlast["output_ids_range"]
    check(start == input_len,
          f"output_ids_range start {start} == input_len {input_len}")
    check(end == input_len + raw_output_ids.shape[0],
          f"output_ids_range end {end} == {input_len + raw_output_ids.shape[0]}")

    # attention_mask length matches input_ids
    check(wlast["attention_mask"].shape[0] == wlast["input_ids"].shape[0],
          "attention_mask length == input_ids length (with output appended)")

    # Decode output for visual inspection
    decoded = tokenizer.decode(raw_output_ids, skip_special_tokens=False)
    print(f"  Output decoded: {decoded[:200]}")

    # ==================================================================
    # 4. Cross-window trajectory consistency
    # ==================================================================
    print("\n" + "=" * 60)
    print("4. CROSS-WINDOW TRAJECTORY CONSISTENCY")
    print("=" * 60)

    for i in range(len(item)):
        check(torch.equal(item[i]["ego_history_xyz"], raw_subseq[i]["ego_history_xyz"].squeeze(0)),
              f"window {i} ego_history_xyz matches raw (squeezed)")

    timestamps = [raw_subseq[i]["timestamp"] for i in range(len(raw_subseq))]
    diffs = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    check(all(d > 0 for d in diffs),
          f"timestamps strictly increasing: {timestamps}, diffs={diffs}")

    # ==================================================================
    # 5. Expert training stage
    # ==================================================================
    print("\n" + "=" * 60)
    print("5. EXPERT TRAINING STAGE")
    print("=" * 60)

    dataset_expert = StreamingDataset(cfg, processor=processor, training_stage="expert")
    item_expert = dataset_expert[0]
    wlast_e = item_expert[-1]

    check("ego_future_xyz" in wlast_e, "expert last window has ego_future_xyz")
    check("ego_future_rot" in wlast_e, "expert last window has ego_future_rot")
    check("labels" not in wlast_e, "expert last window has NO labels")
    check("output_ids_range" not in wlast_e, "expert last window has NO output_ids_range")
    if len(item_expert) > 1:
        check("ego_future_xyz" not in item_expert[0],
              "expert rollout window has NO ego_future_xyz")

    # Expert last window input_ids should NOT include output_token_ids
    _cid_e, _st_e, _sl_e = dataset_expert.data[0]
    _raw_e = torch.load(os.path.join(data_dir, _cid_e, "sliding_window_inputs.pt"),
                        map_location="cpu", weights_only=False)
    raw_last_e = _raw_e[_st_e + _sl_e - 1]
    raw_ids_e = raw_last_e["tokenized_data"]["input_ids"].squeeze(0)
    if len(item_expert) > 1:
        keep_e = _compute_streaming_keep_mask(raw_ids_e)
        check(torch.equal(wlast_e["input_ids"], raw_ids_e[keep_e]),
              "expert last window input_ids has NO output_token_ids appended")

    # ==================================================================
    # 6. Collate function
    # ==================================================================
    print("\n" + "=" * 60)
    print("6. COLLATE FUNCTION")
    print("=" * 60)

    batch = collate_fn([item])
    check(len(batch) == len(item), "collated batch has same number of windows")
    check(batch[0]["input_ids"].shape == (1, item[0]["input_ids"].shape[0]),
          "collate adds batch dim to input_ids")
    check(batch[0]["pixel_values"].shape == item[0]["pixel_values"].shape,
          "collate does NOT add batch dim to pixel_values")
    check(batch[0]["image_grid_thw"].shape == item[0]["image_grid_thw"].shape,
          "collate does NOT add batch dim to image_grid_thw")
    check(batch[0]["is_prefill"] is True,
          "collate preserves non-tensor fields")

    # ==================================================================
    # 7. Random rollout length variation
    # ==================================================================
    print("\n" + "=" * 60)
    print("7. RANDOM ROLLOUT LENGTH VARIATION")
    print("=" * 60)

    cfg_multi = OmegaConf.create({
        "data_dir": data_dir,
        "clip_list": clip_list_path,
        "min_rollout_steps": MIN_ROLLOUT_STEPS,
        "max_rollout_steps": MAX_ROLLOUT_STEPS,
        "samples_per_clip": 20,
    })
    dataset_multi = StreamingDataset(cfg_multi, processor=processor, training_stage="vlm")
    lengths = [len(dataset_multi[i]) for i in range(len(dataset_multi))]
    unique_lengths = set(lengths)
    print(f"  {len(dataset_multi)} items, lengths: {sorted(lengths)}")
    print(f"  Unique lengths: {sorted(unique_lengths)}")
    check(all(MIN_ROLLOUT_STEPS + 1 <= l <= MAX_ROLLOUT_STEPS + 1 for l in lengths),
          f"all lengths in [{MIN_ROLLOUT_STEPS + 1}, {MAX_ROLLOUT_STEPS + 1}]")
    check(len(unique_lengths) > 1,
          f"multiple distinct lengths observed: {sorted(unique_lengths)}")

    # Verify no duplicate (clip_id, start, seq_len)
    seen_ids = set()
    for entry in dataset_multi.data:
        check(entry not in seen_ids,
              f"unique subsequence: clip={entry[0][:8]}..., start={entry[1]}, len={entry[2]}")
        seen_ids.add(entry)

    # ==================================================================
    # 8. EvalStreamingDataset uses fixed rollout length
    # ==================================================================
    print("\n" + "=" * 60)
    print("8. EVAL DATASET — fixed rollout length")
    print("=" * 60)

    cfg_eval = OmegaConf.create({
        "data_dir": data_dir,
        "clip_list": clip_list_path,
        "min_rollout_steps": MIN_ROLLOUT_STEPS,
        "max_rollout_steps": MAX_ROLLOUT_STEPS,
        "samples_per_clip": 5,
    })
    dataset_eval = EvalStreamingDataset(cfg_eval, processor=processor)
    eval_lengths = [len(dataset_eval[i]) for i in range(len(dataset_eval))]
    check(all(l == MAX_ROLLOUT_STEPS + 1 for l in eval_lengths),
          f"all eval items have fixed length {MAX_ROLLOUT_STEPS + 1}: {eval_lengths}")

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
    print("=" * 60)
