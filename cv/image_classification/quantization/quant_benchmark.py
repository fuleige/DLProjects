import csv
import gc
import json
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
import torchvision

try:
    from .quant_types import BenchmarkStats, CompareResult
except ImportError:
    from quant_types import BenchmarkStats, CompareResult


def read_process_rss_mb() -> float | None:
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None

    for line in status_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("VmRSS:"):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        return int(parts[1]) / 1024.0
    return None


@torch.inference_mode()
def benchmark_cpu_inference(model, example_batch, warmup, iters, set_eval_mode=True):
    batch = example_batch.to("cpu")
    if set_eval_mode and hasattr(model, "eval"):
        model.eval()

    gc.collect()
    start_rss_mb = read_process_rss_mb()
    peak_rss_mb = start_rss_mb

    for _ in range(max(0, warmup)):
        model(batch)
        current_rss_mb = read_process_rss_mb()
        if current_rss_mb is not None:
            peak_rss_mb = current_rss_mb if peak_rss_mb is None else max(
                peak_rss_mb,
                current_rss_mb,
            )

    start = time.perf_counter()
    for _ in range(max(1, iters)):
        model(batch)
        current_rss_mb = read_process_rss_mb()
        if current_rss_mb is not None:
            peak_rss_mb = current_rss_mb if peak_rss_mb is None else max(
                peak_rss_mb,
                current_rss_mb,
            )
    end = time.perf_counter()
    rss_delta_mb = None
    if start_rss_mb is not None and peak_rss_mb is not None:
        rss_delta_mb = max(0.0, peak_rss_mb - start_rss_mb)
    return BenchmarkStats(
        latency_ms=(end - start) * 1000 / max(1, iters),
        peak_rss_mb=peak_rss_mb,
        rss_delta_mb=rss_delta_mb,
    )


def benchmark_deploy_inference(model, example_batch, warmup, iters, backend, set_eval_mode=True):
    if backend != "x86_inductor":
        print(
            f"提示: 当前 backend={backend}，脚本不会用 torch.compile(inductor) 生成 deploy 延迟。"
            "此时请主要参考 eager CPU 延迟；如果需要真实 xnnpack/ExecuTorch 部署数据，请在对应运行时里单独测试。"
        )
        return None
    try:
        compiled_model = torch.compile(model, backend="inductor")
        return benchmark_cpu_inference(
            compiled_model,
            example_batch,
            warmup=warmup,
            iters=iters,
            set_eval_mode=set_eval_mode,
        )
    except Exception as exc:
        print(f"警告: torch.compile(inductor) 延迟基准失败，将跳过 deploy 延迟。原因: {exc}")
        return None


def print_compare_table(results: Iterable[CompareResult], topk_name):
    results = list(results)
    baseline_latency = None
    for result in results:
        if result.name == "float32":
            baseline_latency = result.deploy_latency_ms or result.eager_latency_ms
            break

    print("\n结果对比：")
    print(
        f"{'Method':<12} {'Top1':>10} {'%s' % topk_name:>10} {'Eager(ms)':>12} "
        f"{'Deploy(ms)':>12} {'RSSΔ(MB)':>12} {'TrainCUDA':>12} {'Speedup':>10}  Notes"
    )
    print("-" * 124)
    for result in results:
        eager_str = "-" if result.eager_latency_ms is None else f"{result.eager_latency_ms:.2f}"
        deploy_str = "-" if result.deploy_latency_ms is None else f"{result.deploy_latency_ms:.2f}"
        memory_str = (
            "-" if result.benchmark_rss_delta_mb is None else f"{result.benchmark_rss_delta_mb:.1f}"
        )
        train_cuda_str = (
            "-" if result.train_peak_cuda_mb is None else f"{result.train_peak_cuda_mb:.1f}"
        )
        reference_latency = result.deploy_latency_ms or result.eager_latency_ms
        if baseline_latency is None or reference_latency is None:
            speedup_str = "-"
        else:
            speedup_str = f"{baseline_latency / reference_latency:.2f}x"
        print(
            f"{result.name:<12} {result.top1:>10.4f} {result.topk:>10.4f} "
            f"{eager_str:>12} {deploy_str:>12} {memory_str:>12} {train_cuda_str:>12} "
            f"{speedup_str:>10}  {result.notes}"
        )


def collect_runtime_info():
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": getattr(torchvision, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    try:
        import torchao

        info["torchao"] = getattr(torchao, "__version__", "unknown")
    except Exception as exc:
        info["torchao"] = f"IMPORT_ERROR: {exc}"

    try:
        import executorch

        info["executorch"] = getattr(executorch, "__version__", "unknown")
    except Exception as exc:
        info["executorch"] = f"IMPORT_ERROR: {exc}"

    return info


def build_result_rows(results: Iterable[CompareResult]):
    rows = []
    baseline_latency = None
    for result in results:
        if result.name == "float32":
            baseline_latency = result.deploy_latency_ms or result.eager_latency_ms
            break

    for result in results:
        reference_latency = result.deploy_latency_ms or result.eager_latency_ms
        speedup = None
        if baseline_latency is not None and reference_latency is not None:
            speedup = baseline_latency / reference_latency
        rows.append(
            {
                "method": result.name,
                "top1": round(result.top1, 6),
                "topk": round(result.topk, 6),
                "eager_latency_ms": None
                if result.eager_latency_ms is None
                else round(result.eager_latency_ms, 4),
                "deploy_latency_ms": None
                if result.deploy_latency_ms is None
                else round(result.deploy_latency_ms, 4),
                "speedup_vs_float": None if speedup is None else round(speedup, 4),
                "benchmark_peak_rss_mb": None
                if result.benchmark_peak_rss_mb is None
                else round(result.benchmark_peak_rss_mb, 2),
                "benchmark_rss_delta_mb": None
                if result.benchmark_rss_delta_mb is None
                else round(result.benchmark_rss_delta_mb, 2),
                "train_peak_cuda_mb": None
                if result.train_peak_cuda_mb is None
                else round(result.train_peak_cuda_mb, 2),
                "notes": result.notes,
            }
        )
    return rows


def write_compare_artifacts(output_dir: Path, args, results: Iterable[CompareResult], topk_name: str):
    results = list(results)
    rows = build_result_rows(results)
    runtime_info = collect_runtime_info()
    payload = {
        "args": vars(args),
        "runtime": runtime_info,
        "topk_name": topk_name,
        "results": rows,
        "memory_note": (
            "benchmark_peak_rss_mb 和 benchmark_rss_delta_mb 来自当前 Python 进程在推理基准阶段的 "
            "CPU RSS 采样；train_peak_cuda_mb 是训练或微调阶段的峰值 CUDA 显存。"
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark_results.json"
    csv_path = output_dir / "benchmark_results.csv"
    md_path = output_dir / "benchmark_results.md"

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fieldnames = list(rows[0].keys()) if rows else [
        "method",
        "top1",
        "topk",
        "eager_latency_ms",
        "deploy_latency_ms",
        "speedup_vs_float",
        "benchmark_peak_rss_mb",
        "benchmark_rss_delta_mb",
        "train_peak_cuda_mb",
        "notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# torchao 量化实验结果",
        "",
        f"- 生成时间：`{runtime_info['timestamp']}`",
        f"- 命令：`{runtime_info['command']}`",
        f"- 模型：`{args.model_name}`",
        f"- 数据集：`{args.dataset_type}`",
        f"- backend：`{args.backend}`",
        f"- `torch / torchvision / torchao`：`{runtime_info['torch']} / {runtime_info['torchvision']} / {runtime_info['torchao']}`",
        f"- CUDA：`{runtime_info['cuda_device_name'] or 'CPU only'}`",
        "",
        "| Method | Top1 | "
        + topk_name
        + " | Eager(ms) | Deploy(ms) | Speedup | Peak RSS(MB) | RSS Delta(MB) | Train Peak CUDA(MB) | Notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    "-" if row["top1"] is None else f"{row['top1']:.4f}",
                    "-" if row["topk"] is None else f"{row['topk']:.4f}",
                    "-"
                    if row["eager_latency_ms"] is None
                    else f"{row['eager_latency_ms']:.2f}",
                    "-"
                    if row["deploy_latency_ms"] is None
                    else f"{row['deploy_latency_ms']:.2f}",
                    "-"
                    if row["speedup_vs_float"] is None
                    else f"{row['speedup_vs_float']:.2f}x",
                    "-"
                    if row["benchmark_peak_rss_mb"] is None
                    else f"{row['benchmark_peak_rss_mb']:.1f}",
                    "-"
                    if row["benchmark_rss_delta_mb"] is None
                    else f"{row['benchmark_rss_delta_mb']:.1f}",
                    "-"
                    if row["train_peak_cuda_mb"] is None
                    else f"{row['train_peak_cuda_mb']:.1f}",
                    str(row["notes"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 指标说明",
            "",
            "- `Peak RSS(MB)` / `RSS Delta(MB)` 是推理基准期间采样到的 CPU 常驻内存。",
            "- `Train Peak CUDA(MB)` 记录训练或微调阶段的峰值 GPU 显存；PTQ 没有训练阶段，因此该列为空。",
            "- `Deploy(ms)` 优先反映 `torch.compile(inductor)` 之后的部署延迟，适合作为最终速度对比主指标。",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n已写出实验结果文件：")
    print(f"- JSON: {json_path}")
    print(f"- CSV : {csv_path}")
    print(f"- MD  : {md_path}")


def get_deploy_note(backend: str) -> str:
    if backend == "x86_inductor":
        return "deploy 延迟来自 torch.compile(inductor)"
    return "未提供 deploy 延迟；当前脚本对该 backend 主要记录 eager CPU 延迟"
