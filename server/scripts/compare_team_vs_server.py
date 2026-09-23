"""모델팀 추론 코드와 서버 추론 경로를 안경 영상 50개로 비교한다.

    team    모델팀이 보낸 predict_segment 그대로.
            fps_step 을 가정해 raw 를 ×N 업샘플 -> build_features -> 60
    server  서버 word_segment_service.build_model_input.
            촬영 시각 기준 30fps 복원 -> build_features -> 60 (half-pixel)

두 방식은 프레임이 빠졌을 때만 갈린다. 그래서 드롭 조건을 셋 둔다.

    none     30fps 전 프레임 (두 방식이 거의 같아야 정상)
    regular  3장 중 1장 (모델팀 fps_step=3 가정이 정확히 맞는 경우)
    random   간격 2~7장 무작위 (실제 서버에 가까운 경우, 시드 3개 평균)

그 위에 fold0 단독과 fold0~4 앙상블(확률 평균)을 각각 돌린다.
안경 영상은 학습에 쓰지 않았으므로 앙상블 결과가 부풀려지지 않는다.

    python scripts/compare_team_vs_server.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.feature_service import build_features  # noqa: E402
from app.services.recognition_model import (  # noqa: E402
    FEATURE_DIM,
    SEQUENCE_LENGTH,
    load_recognition_model,
    make_predictor,
)
from app.services.word_segment_service import build_model_input  # noqa: E402

CONF_T, MARGIN_T = 0.5, 0.15


def team_input(raw: np.ndarray, fps_step: int) -> np.ndarray:
    """모델팀 predict_segment 의 ①② 를 글자 그대로 옮긴 것."""
    import tensorflow as tf

    if fps_step > 1:
        raw = tf.image.resize(raw[..., None], [len(raw) * fps_step, 411],
                              method="bilinear")[..., 0].numpy()
    feat = build_features(raw)
    return tf.image.resize(feat[..., None], [SEQUENCE_LENGTH, FEATURE_DIM],
                           method="bilinear")[..., 0].numpy().astype(np.float32)


def drop_indices(n: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode == "none":
        return np.arange(n)
    if mode == "regular":
        return np.arange(0, n, 3)
    idx, i = [], 0
    while i < n:
        idx.append(i)
        i += int(rng.integers(2, 8))
    return np.array(idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT.parent / "data" / "keypoint_cache_glasses"))
    ap.add_argument("--out", default="compare_team_vs_server.csv")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.cache, "*.npz")))
    predictors = [make_predictor(load_recognition_model(ROOT / "models" / f"model_fold{k}.keras"))
                  for k in range(5)]
    for p in predictors:
        p(np.zeros((1, SEQUENCE_LENGTH, FEATURE_DIM), np.float32))

    rows = []
    for f in files:
        d = np.load(f)
        raw, ts = d["frames"].astype(np.float32), d["timestamps_ms"].astype(np.float64)
        label = int(str(d["word_id"])[4:]) - 1
        for mode, seeds in (("none", [0]), ("regular", [0]), ("random", [1, 2, 3])):
            for seed in seeds:
                keep = drop_indices(len(raw), mode, np.random.default_rng(seed))
                r, t = raw[keep], ts[keep]
                step = 1 if mode == "none" else max(1, int(round(np.mean(np.diff(keep)))))
                inputs = {
                    "team": team_input(r, step),
                    "server": build_model_input(r.tolist(), t.tolist())[0],
                }
                for method, x in inputs.items():
                    probs = np.stack([np.asarray(p(x[None]))[0] for p in predictors])
                    for model_name, prob in (("fold0", probs[0]), ("ensemble", probs.mean(0))):
                        order = np.argsort(-prob)
                        top, second = float(prob[order[0]]), float(prob[order[1]])
                        rows.append({
                            "file": Path(f).stem, "drop": mode, "seed": seed, "kept": len(keep),
                            "method": method, "model": model_name, "label": label,
                            "pred": int(order[0]), "correct": int(order[0] == label),
                            "top": top, "margin": top - second,
                            "p_true": float(prob[label]),
                            "accepted": int(top >= CONF_T and top - second >= MARGIN_T),
                        })

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'drop':8} {'method':7} {'model':9} {'정답률':>6} {'채택':>6} {'채택중정답':>8} "
          f"{'오답확신중앙':>9} {'오답>0.5':>8} {'정답<0.5':>8}")
    for mode in ("none", "regular", "random"):
        for method in ("team", "server"):
            for model_name in ("fold0", "ensemble"):
                g = [r for r in rows if r["drop"] == mode and r["method"] == method
                     and r["model"] == model_name]
                acc = np.mean([r["correct"] for r in g])
                accepted = [r for r in g if r["accepted"]]
                wrong = [r["top"] for r in g if not r["correct"]]
                right = [r["top"] for r in g if r["correct"]]
                print(f"{mode:8} {method:7} {model_name:9} {acc:6.1%} {len(accepted)/len(g):6.1%} "
                      f"{(np.mean([r['correct'] for r in accepted]) if accepted else float('nan')):8.1%} "
                      f"{(np.median(wrong) if wrong else float('nan')):9.3f} "
                      f"{(np.mean([w > CONF_T for w in wrong]) if wrong else float('nan')):8.1%} "
                      f"{(np.mean([c < CONF_T for c in right]) if right else float('nan')):8.1%}")


if __name__ == "__main__":
    main()
