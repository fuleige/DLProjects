import gc
from pathlib import Path

import torch

try:
    from .quant_args import parse_args, print_supported_models
    from .quant_benchmark import (
        benchmark_cpu_inference,
        benchmark_deploy_inference,
        get_deploy_note,
        print_compare_table,
        write_compare_artifacts,
    )
    from .quant_core import (
        build_datasets,
        build_experiment_tag,
        build_float_model_from_checkpoint,
        build_loader,
        build_transforms,
        resolve_model_recipe,
        set_seed,
        train_float_model,
    )
    from .quant_pt2e import run_ptq, run_qat
    from .quant_types import CompareResult, MODEL_SPECS
except ImportError:
    from quant_args import parse_args, print_supported_models
    from quant_benchmark import (
        benchmark_cpu_inference,
        benchmark_deploy_inference,
        get_deploy_note,
        print_compare_table,
        write_compare_artifacts,
    )
    from quant_core import (
        build_datasets,
        build_experiment_tag,
        build_float_model_from_checkpoint,
        build_loader,
        build_transforms,
        resolve_model_recipe,
        set_seed,
        train_float_model,
    )
    from quant_pt2e import run_ptq, run_qat
    from quant_types import CompareResult, MODEL_SPECS

"""
torchao 图像分类量化示例：
1. 先训练或加载一个浮点分类模型。
2. 用 PT2E 路线完成 PTQ（export -> prepare_pt2e -> calibrate -> convert_pt2e）。
3. 用 PT2E 路线完成 QAT（export -> prepare_qat_pt2e -> fine-tune -> convert_pt2e）。
4. 输出 float / PTQ / QAT 的精度、部署速度和资源占用对比。

这里故意使用 torchvision 的经典 CNN，而不是任意 timm 模型。原因是：
- torchao 当前图像分类 PT2E 教程主要围绕标准 CNN 展开；
- ResNet / MobileNet 这类 Conv2d + BatchNorm + Linear 模型更适合作为 static int8 教学模板；
- 这样能把“量化流程本身”讲清楚，而不是先被模型导出兼容性问题干扰。
"""


def summarize_context(args, train_dataset, calib_dataset, val_dataset, num_classes, output_dir):
    spec = MODEL_SPECS[args.model_name]
    print(f"数据集类型: {args.dataset_type}")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"校准样本数: {len(calib_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    print(f"类别数: {num_classes}")
    print(f"模型: {spec.alias}")
    print(f"说明: {spec.summary}")
    print(f"建议场景: {spec.recommended_for}")
    print(f"量化 backend: {args.backend}")
    print(f"实验输出目录: {output_dir}")


def build_float_result(args, checkpoint_path, metrics, train_peak_cuda_mb, num_classes, val_loader):
    float_model = build_float_model_from_checkpoint(
        args,
        num_classes,
        checkpoint_path,
        torch.device("cpu"),
    )
    example_batch = next(iter(val_loader))[0]
    eager_benchmark = benchmark_cpu_inference(
        float_model,
        example_batch,
        warmup=args.benchmark_warmup,
        iters=args.benchmark_iters,
    )
    deploy_benchmark = benchmark_deploy_inference(
        float_model,
        example_batch,
        warmup=args.benchmark_warmup,
        iters=args.benchmark_iters,
        backend=args.backend,
    )
    preferred_benchmark = deploy_benchmark or eager_benchmark
    deploy_note = get_deploy_note(args.backend)
    result = CompareResult(
        name="float32",
        top1=metrics.top1,
        topk=metrics.topk,
        eager_latency_ms=eager_benchmark.latency_ms,
        deploy_latency_ms=None if deploy_benchmark is None else deploy_benchmark.latency_ms,
        benchmark_peak_rss_mb=preferred_benchmark.peak_rss_mb,
        benchmark_rss_delta_mb=preferred_benchmark.rss_delta_mb,
        train_peak_cuda_mb=train_peak_cuda_mb,
        notes=f"浮点基线，{deploy_note}",
    )
    del float_model
    gc.collect()
    return result


def build_ptq_result(args, metrics, eager_benchmark, deploy_benchmark):
    preferred_benchmark = deploy_benchmark or eager_benchmark
    return CompareResult(
        name="ptq_int8",
        top1=metrics.top1,
        topk=metrics.topk,
        eager_latency_ms=eager_benchmark.latency_ms,
        deploy_latency_ms=None if deploy_benchmark is None else deploy_benchmark.latency_ms,
        benchmark_peak_rss_mb=preferred_benchmark.peak_rss_mb,
        benchmark_rss_delta_mb=preferred_benchmark.rss_delta_mb,
        train_peak_cuda_mb=None,
        notes=f"无需再训练，只做校准；{get_deploy_note(args.backend)}",
    )


def build_qat_result(args, metrics, eager_benchmark, deploy_benchmark, train_peak_cuda_mb, checkpoint_path):
    preferred_benchmark = deploy_benchmark or eager_benchmark
    return CompareResult(
        name="qat_int8",
        top1=metrics.top1,
        topk=metrics.topk,
        eager_latency_ms=eager_benchmark.latency_ms,
        deploy_latency_ms=None if deploy_benchmark is None else deploy_benchmark.latency_ms,
        benchmark_peak_rss_mb=preferred_benchmark.peak_rss_mb,
        benchmark_rss_delta_mb=preferred_benchmark.rss_delta_mb,
        train_peak_cuda_mb=train_peak_cuda_mb,
        notes=(
            "QAT 微调默认优先走 GPU；最终量化验证来自 CPU，"
            f"{get_deploy_note(args.backend)}，"
            f"prepared ckpt: {checkpoint_path.name}"
        ),
    )


def finalize_results(args, output_dir, results, topk_name):
    print_compare_table(results, topk_name)
    write_compare_artifacts(output_dir, args, results, topk_name)


def main():
    args = parse_args()
    if args.list_models:
        print_supported_models()
        return

    set_seed(args.seed)

    _, _, mean, std, interpolation = resolve_model_recipe(
        args.model_name,
        pretrained=args.pretrained,
    )
    train_transform, eval_transform = build_transforms(mean, std, interpolation)
    train_dataset, calib_dataset, val_dataset, class_names = build_datasets(
        args,
        train_transform,
        eval_transform,
    )
    num_classes = len(class_names)
    topk = min(5, num_classes)
    topk_name = f"top{topk}"

    train_loader = build_loader(train_dataset, args.batch_size, args.num_workers, shuffle=True)
    calib_loader = build_loader(calib_dataset, args.batch_size, args.num_workers, shuffle=False)
    val_loader = build_loader(val_dataset, args.batch_size, args.num_workers, shuffle=False)

    experiment_tag = build_experiment_tag(args)
    output_dir = Path(args.output_dir) / args.model_name / experiment_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    summarize_context(args, train_dataset, calib_dataset, val_dataset, num_classes, output_dir)

    checkpoint_path, float_metrics, float_train_peak_cuda_mb = train_float_model(
        args,
        train_loader,
        val_loader,
        num_classes,
        class_names,
        output_dir,
    )
    float_result = build_float_result(
        args,
        checkpoint_path,
        float_metrics,
        float_train_peak_cuda_mb,
        num_classes,
        val_loader,
    )

    if args.mode == "train_float":
        finalize_results(args, output_dir, [float_result], topk_name)
        return

    if args.mode == "ptq":
        _, ptq_metrics, ptq_eager_benchmark, ptq_deploy_benchmark = run_ptq(
            args,
            calib_loader,
            val_loader,
            num_classes,
            checkpoint_path,
        )
        finalize_results(
            args,
            output_dir,
            [
                float_result,
                build_ptq_result(args, ptq_metrics, ptq_eager_benchmark, ptq_deploy_benchmark),
            ],
            topk_name,
        )
        return

    if args.mode == "qat":
        (
            _,
            qat_metrics,
            qat_eager_benchmark,
            qat_deploy_benchmark,
            qat_train_peak_cuda_mb,
            qat_checkpoint_path,
        ) = run_qat(
            args,
            train_loader,
            val_loader,
            num_classes,
            checkpoint_path,
            output_dir,
        )
        finalize_results(
            args,
            output_dir,
            [
                float_result,
                build_qat_result(
                    args,
                    qat_metrics,
                    qat_eager_benchmark,
                    qat_deploy_benchmark,
                    qat_train_peak_cuda_mb,
                    qat_checkpoint_path,
                ),
            ],
            topk_name,
        )
        return

    _, ptq_metrics, ptq_eager_benchmark, ptq_deploy_benchmark = run_ptq(
        args,
        calib_loader,
        val_loader,
        num_classes,
        checkpoint_path,
    )
    (
        _,
        qat_metrics,
        qat_eager_benchmark,
        qat_deploy_benchmark,
        qat_train_peak_cuda_mb,
        qat_checkpoint_path,
    ) = run_qat(
        args,
        train_loader,
        val_loader,
        num_classes,
        checkpoint_path,
        output_dir,
    )
    finalize_results(
        args,
        output_dir,
        [
            float_result,
            build_ptq_result(args, ptq_metrics, ptq_eager_benchmark, ptq_deploy_benchmark),
            build_qat_result(
                args,
                qat_metrics,
                qat_eager_benchmark,
                qat_deploy_benchmark,
                qat_train_peak_cuda_mb,
                qat_checkpoint_path,
            ),
        ],
        topk_name,
    )


if __name__ == "__main__":
    main()
