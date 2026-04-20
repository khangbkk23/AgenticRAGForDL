from dataclasses import dataclass
from typing import List, Optional
import yaml


# LLM
@dataclass
class LLMConfig:
    provider: str
    base_url: str
    model_name: str
    temperature: float
    max_tokens: int
    context_length: int


# Agent
@dataclass
class AgentConfig:
    max_steps: int
    allowed_actions: List[str]
    early_stopping: bool


# Retrieval
@dataclass
class DenseRetrievalConfig:
    embedding_model: str
    vector_store_path: str
    top_k: int


@dataclass
class RerankerConfig:
    enabled: bool
    model: Optional[str]
    top_k: int


@dataclass
class RetrievalConfig:
    dense: DenseRetrievalConfig
    reranker: RerankerConfig


# Tools
@dataclass
class ProfilerConfig:
    enabled: bool
    warmup_steps: int
    active_steps: int


@dataclass
class HardwareLimits:
    max_ram_gb: int
    max_vram_gb: int
    cuda_visible_devices: str


@dataclass
class CodeExecutorConfig:
    timeout: int
    isolation: str
    hardware_limits: HardwareLimits


@dataclass
class ToolsConfig:
    profiler: ProfilerConfig
    code_executor: CodeExecutorConfig


# Optimization
@dataclass
class TargetMetrics:
    latency_reduction_pct: int
    vram_reduction_pct: int


@dataclass
class TechniquesConfig:
    enable_amp: bool
    enable_compile: bool
    enable_tensorrt: bool


@dataclass
class OptimizationConfig:
    target_metrics: TargetMetrics
    techniques: TechniquesConfig

# Logging
@dataclass
class LoggingConfig:
    level: str
    log_file: str


@dataclass
class Config:
    llm: LLMConfig
    agent: AgentConfig
    retrieval: RetrievalConfig
    tools: ToolsConfig
    optimization: OptimizationConfig
    logging: LoggingConfig


def load_config(path: str = "configs/config.yaml") -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # LLM
    llm = LLMConfig(**raw["llm"])

    # Agent
    agent = AgentConfig(**raw["agent"])

    # Retrieval
    dense = DenseRetrievalConfig(**raw["retrieval"]["dense"])
    reranker = RerankerConfig(**raw["retrieval"]["reranker"])
    retrieval = RetrievalConfig(dense=dense, reranker=reranker)

    # Tools
    profiler = ProfilerConfig(**raw["tools"]["profiler"])

    hw = HardwareLimits(**raw["tools"]["code_executor"]["hardware_limits"])
    executor = CodeExecutorConfig(
        timeout=raw["tools"]["code_executor"]["timeout"],
        isolation=raw["tools"]["code_executor"]["isolation"],
        hardware_limits=hw,
    )

    tools = ToolsConfig(
        profiler=profiler,
        code_executor=executor,
    )

    # Optimization
    target_metrics = TargetMetrics(**raw["optimization"]["target_metrics"])
    techniques = TechniquesConfig(**raw["optimization"]["techniques"])
    optimization = OptimizationConfig(
        target_metrics=target_metrics,
        techniques=techniques,
    )

    # Logging
    logging = LoggingConfig(**raw["logging"])

    config = Config(
        llm=llm,
        agent=agent,
        retrieval=retrieval,
        tools=tools,
        optimization=optimization,
        logging=logging,
    )

    _validate_config(config)
    return config


def _validate_config(config: Config):
    assert config.agent.max_steps > 0, "max_steps must be > 0"
    assert config.llm.max_tokens <= config.llm.context_length, \
        "max_tokens must be <= context_length"

    if config.retrieval.reranker.enabled:
        assert config.retrieval.reranker.model is not None, \
            "Reranker enabled but no model specified"

    assert config.tools.code_executor.isolation in ["docker", "subprocess"], \
        "Invalid isolation type"

    assert config.tools.code_executor.hardware_limits.max_ram_gb > 0
    assert config.tools.code_executor.hardware_limits.max_vram_gb > 0