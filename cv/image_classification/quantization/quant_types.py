from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    release_year: int
    summary: str
    recommended_for: str


@dataclass
class MetricResult:
    loss: float
    top1: float
    topk: float


@dataclass
class BenchmarkStats:
    latency_ms: float
    peak_rss_mb: float | None
    rss_delta_mb: float | None


@dataclass
class CompareResult:
    name: str
    top1: float
    topk: float
    eager_latency_ms: float | None
    deploy_latency_ms: float | None
    benchmark_peak_rss_mb: float | None
    benchmark_rss_delta_mb: float | None
    train_peak_cuda_mb: float | None
    notes: str


MODEL_SPECS = {
    "resnet18": ModelSpec(
        alias="resnet18",
        release_year=2015,
        summary="结构简单，量化图里最容易看懂，适合作为 PTQ/QAT 第一套模板。",
        recommended_for="先把 PT2E 量化流程跑通，再迁移到自己的分类模型。",
    ),
    "mobilenet_v3_small": ModelSpec(
        alias="mobilenet_v3_small",
        release_year=2019,
        summary="更偏轻量部署，量化后更接近真实移动端分类场景。",
        recommended_for="希望同时关注量化精度和推理开销。",
    ),
}
