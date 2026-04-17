import copy
import gc
import inspect

import torch
from torch import nn

try:
    from .quant_benchmark import benchmark_cpu_inference, benchmark_deploy_inference
    from .quant_core import (
        build_float_model_from_checkpoint,
        evaluate,
        format_metrics,
        get_device,
        get_peak_cuda_memory_mb,
        train_one_epoch,
    )
except ImportError:
    from quant_benchmark import benchmark_cpu_inference, benchmark_deploy_inference
    from quant_core import (
        build_float_model_from_checkpoint,
        evaluate,
        format_metrics,
        get_device,
        get_peak_cuda_memory_mb,
        train_one_epoch,
    )


def export_with_dynamic_batch(model, example_inputs, min_batch=None, max_batch=None):
    """Export a model with a dynamic batch axis for the PT2E flow."""
    inferred_min_batch = min_batch
    if inferred_min_batch is None and example_inputs and isinstance(example_inputs[0], torch.Tensor):
        inferred_min_batch = int(example_inputs[0].shape[0])
    dynamic_shapes = tuple(
        {
            0: torch.export.Dim(
                "batch",
                min=inferred_min_batch,
                max=max_batch,
            )
        }
        if i == 0
        else None
        for i in range(len(example_inputs))
    )
    return torch.export.export(
        model,
        example_inputs,
        dynamic_shapes=dynamic_shapes,
    ).module()


def import_pt2e_apis():
    try:
        import torchao
        from torchao.quantization.pt2e.quantize_pt2e import (
            convert_pt2e,
            prepare_pt2e,
            prepare_qat_pt2e,
        )
    except Exception as exc:
        raise RuntimeError(
            "当前环境中的 torchao 无法正常导入。常见原因是 torch 与 torchao 版本不匹配，"
            "或者 torchao 安装不完整。请先校对当前虚拟环境里的 torch / torchao 版本。"
        ) from exc
    return torchao, prepare_pt2e, prepare_qat_pt2e, convert_pt2e


def build_quantizer(backend, is_qat):
    """Create a backend-specific PT2E quantizer.

    This example line only covers static int8 image classification, so the
    x86 branch explicitly disables dynamic quantization when the API supports it.
    """
    if backend == "xnnpack":
        try:
            from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
                XNNPACKQuantizer,
                get_symmetric_quantization_config,
            )
        except ImportError as exc:
            raise RuntimeError(
                "xnnpack backend 需要额外安装 executorch，例如 `pip install executorch`。"
            ) from exc
        kwargs = {}
        signature = inspect.signature(get_symmetric_quantization_config)
        if "is_qat" in signature.parameters:
            kwargs["is_qat"] = is_qat
        if "is_per_channel" in signature.parameters:
            kwargs["is_per_channel"] = True
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(get_symmetric_quantization_config(**kwargs))
        return quantizer

    if backend == "x86_inductor":
        from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
            X86InductorQuantizer,
            get_default_x86_inductor_quantization_config,
        )

        kwargs = {}
        signature = inspect.signature(get_default_x86_inductor_quantization_config)
        if "is_qat" in signature.parameters:
            kwargs["is_qat"] = is_qat
        if "is_dynamic" in signature.parameters:
            kwargs["is_dynamic"] = False

        quantizer = X86InductorQuantizer()
        quantizer.set_global(get_default_x86_inductor_quantization_config(**kwargs))
        return quantizer

    raise ValueError(f"暂不支持的 backend: {backend}")


def maybe_move_exported_model_to_eval(torchao_module, model):
    move_to_eval = getattr(torchao_module.quantization.pt2e, "move_exported_model_to_eval", None)
    if callable(move_to_eval):
        move_to_eval(model)


def disable_observer_if_supported(torchao_module, model):
    disable_observer = getattr(torchao_module.quantization.pt2e, "disable_observer", None)
    if callable(disable_observer):
        model.apply(disable_observer)


def freeze_bn_stats_in_exported_graph(model):
    """Freeze BatchNorm running stats inside an exported QAT graph."""
    batch_norm_targets = {
        torch.ops.aten._native_batch_norm_legit.default,
    }
    if hasattr(torch.ops.aten, "cudnn_batch_norm"):
        batch_norm_targets.add(torch.ops.aten.cudnn_batch_norm.default)

    changed = False
    for node in model.graph.nodes:
        if node.target not in batch_norm_targets:
            continue
        node_args = list(node.args)
        if len(node_args) <= 5:
            continue
        node_args[5] = False
        node.args = tuple(node_args)
        changed = True

    if changed:
        model.graph.lint()
        model.recompile()


def calibrate(prepared_model, dataloader, max_batches):
    """Run PTQ calibration on representative CPU inputs."""
    with torch.inference_mode():
        for batch_idx, (images, _) in enumerate(dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            prepared_model(images.to("cpu"))


def run_ptq(args, calib_loader, val_loader, num_classes, checkpoint_path):
    """Run the PTQ branch: float -> export -> prepare -> calibration -> convert."""
    torchao, prepare_pt2e, _, convert_pt2e = import_pt2e_apis()
    float_model = build_float_model_from_checkpoint(
        args,
        num_classes,
        checkpoint_path,
        torch.device("cpu"),
    )
    example_inputs = (torch.randn(2, 3, 224, 224),)

    print(f"\n开始 PTQ，backend: {args.backend}")
    # PTQ starts from a float checkpoint on CPU, because this example measures
    # final quantized accuracy and deploy latency with a CPU deployment target.
    exported_model = export_with_dynamic_batch(
        float_model,
        example_inputs,
        min_batch=1,
        max_batch=args.batch_size,
    )
    quantizer = build_quantizer(args.backend, is_qat=False)
    prepared_model = prepare_pt2e(exported_model, quantizer)

    # Calibration only collects activation statistics. It does not update weights.
    calibrate(prepared_model, calib_loader, args.calib_batches)
    quantized_model = convert_pt2e(prepared_model)
    maybe_move_exported_model_to_eval(torchao, quantized_model)

    criterion = nn.CrossEntropyLoss()
    topk = min(5, num_classes)
    metrics = evaluate(
        quantized_model,
        val_loader,
        criterion,
        torch.device("cpu"),
        topk,
        max_batches=args.max_val_batches,
        set_eval_mode=False,
    )

    example_batch = next(iter(val_loader))[0]
    eager_benchmark = benchmark_cpu_inference(
        quantized_model,
        example_batch,
        warmup=args.benchmark_warmup,
        iters=args.benchmark_iters,
        set_eval_mode=False,
    )
    deploy_benchmark = benchmark_deploy_inference(
        quantized_model,
        example_batch,
        warmup=args.benchmark_warmup,
        iters=args.benchmark_iters,
        backend=args.backend,
        set_eval_mode=False,
    )
    return quantized_model, metrics, eager_benchmark, deploy_benchmark


def run_qat(args, train_loader, val_loader, num_classes, checkpoint_path, output_dir):
    """Run the QAT branch: float -> export -> prepare_qat -> fine-tune -> convert."""
    torchao, _, prepare_qat_pt2e, convert_pt2e = import_pt2e_apis()
    qat_device = get_device(args.qat_device, "qat-device")
    float_model = build_float_model_from_checkpoint(
        args,
        num_classes,
        checkpoint_path,
        qat_device,
        train_mode=True,
    )
    if qat_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(qat_device)

    example_inputs = (torch.randn(2, 3, 224, 224, device=qat_device),)

    print(f"\n开始 QAT，backend: {args.backend}，微调设备: {qat_device}")
    if args.batch_size < 2:
        raise ValueError("QAT 训练图当前要求 batch size >= 2，请调整 --batch-size。")
    exported_model = export_with_dynamic_batch(
        float_model,
        example_inputs,
        min_batch=2,
        max_batch=args.batch_size,
    )
    quantizer = build_quantizer(args.backend, is_qat=True)
    # After prepare_qat_pt2e, the model is no longer a plain float model:
    # it becomes a training-ready graph that can simulate quantization error.
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)
    del float_model
    gc.collect()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        prepared_model.parameters(),
        lr=args.qat_learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.qat_epochs),
    )

    topk = min(5, num_classes)
    topk_name = f"top{topk}"
    best_top1 = -1.0
    best_prepared_state = copy.deepcopy(prepared_model.state_dict())
    observer_disabled = False
    bn_frozen = False
    train_peak_cuda_mb = None
    qat_checkpoint_path = output_dir / "qat_prepared_best.pth"

    for epoch in range(args.qat_epochs):
        # The prepared graph keeps its own train/eval semantics, so we avoid
        # toggling mode again inside train_one_epoch/evaluate.
        train_metrics = train_one_epoch(
            prepared_model,
            train_loader,
            criterion,
            optimizer,
            qat_device,
            topk,
            max_batches=args.max_train_batches,
            min_batch_size=2,
            set_training_mode=False,
        )
        scheduler.step()
        train_peak_cuda_mb = get_peak_cuda_memory_mb(qat_device)

        current_epoch = epoch + 1
        if not observer_disabled and current_epoch >= max(1, args.disable_observer_epoch):
            disable_observer_if_supported(torchao, prepared_model)
            observer_disabled = True
            print(f"Epoch {epoch + 1}: 已关闭 observer。")

        if not bn_frozen and current_epoch >= max(1, args.freeze_bn_epoch):
            freeze_bn_stats_in_exported_graph(prepared_model)
            bn_frozen = True
            print(f"Epoch {epoch + 1}: 已冻结 BatchNorm 统计量。")

        proxy_val_metrics = evaluate(
            prepared_model,
            val_loader,
            criterion,
            qat_device,
            topk,
            max_batches=args.max_val_batches,
            min_batch_size=2,
            set_eval_mode=False,
        )

        print(f"\nQAT Epoch [{epoch + 1}/{args.qat_epochs}]")
        print(format_metrics("Train", train_metrics, topk_name))
        print(format_metrics("Val(FakeQ)", proxy_val_metrics, topk_name))

        if proxy_val_metrics.top1 > best_top1:
            best_top1 = proxy_val_metrics.top1
            best_prepared_state = copy.deepcopy(prepared_model.state_dict())
            torch.save(best_prepared_state, qat_checkpoint_path)
            print(f"保存新的 QAT prepared checkpoint 到: {qat_checkpoint_path}")

    prepared_model.load_state_dict(best_prepared_state)
    if qat_device.type == "cuda":
        prepared_model = prepared_model.to("cpu")
        torch.cuda.empty_cache()
    else:
        prepared_model = prepared_model.to("cpu")
    # Final conversion and benchmark are both moved back to CPU so the measured
    # quantized model matches the deployment target discussed in the docs.
    quantized_model = convert_pt2e(prepared_model)
    maybe_move_exported_model_to_eval(torchao, quantized_model)

    final_metrics = evaluate(
        quantized_model,
        val_loader,
        criterion,
        torch.device("cpu"),
        topk,
        max_batches=args.max_val_batches,
        set_eval_mode=False,
    )
    print(format_metrics("Final CPU Quant", final_metrics, topk_name))
    example_batch = next(iter(val_loader))[0]
    eager_benchmark = benchmark_cpu_inference(
        quantized_model,
        example_batch,
        warmup=args.benchmark_warmup,
        iters=args.benchmark_iters,
        set_eval_mode=False,
    )
    deploy_benchmark = benchmark_deploy_inference(
        quantized_model,
        example_batch,
        warmup=args.benchmark_warmup,
        iters=args.benchmark_iters,
        backend=args.backend,
        set_eval_mode=False,
    )
    return (
        quantized_model,
        final_metrics,
        eager_benchmark,
        deploy_benchmark,
        train_peak_cuda_mb,
        qat_checkpoint_path,
    )
