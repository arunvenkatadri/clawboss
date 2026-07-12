"""Edge-reduce integration — supervised hybrid local/cloud LLM routing.

Wraps edge-reduce-llm's RouterEngine so every LLM call goes through
AgentHandler supervision (budgets, timeouts, audit, cost attribution)
while edge-reduce handles the local/cloud routing decision.

Usage:
    from agenthandler.edge_reduce import EdgeReduceLLM

    llm = EdgeReduceLLM(
        supervisor=sv,
        local_base_url="http://localhost:11434/v1",
        cloud_api_key="sk-ant-...",
    )

    # Use as the LLM for pipelines, decisions, skill builders
    pipeline.add_llm_decision(llm, prompt_template="...")
    builder = PipelineBuilder(llm, tools, mgr)
    skill = await SkillBuilder(llm).create("...")

Or with a pre-configured RouterEngine:
    from localreduce.router.engine import RouterEngine
    llm = EdgeReduceLLM.from_engine(engine, supervisor=sv)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, Optional

from .observe import PricingTable
from .supervisor import Supervisor


@dataclass
class EdgeReduceResult:
    """Result from an edge-reduce routed LLM call."""

    text: str
    route: str
    model: str = ""
    confidence: Optional[int] = None
    tokens_local: int = 0
    tokens_cloud: int = 0
    cost_usd: float = 0.0
    latency_local_ms: float = 0.0
    latency_cloud_ms: float = 0.0
    rule_name: str = ""


class EdgeReduceLLM:
    """Supervised hybrid LLM that routes between local and cloud models.

    Combines edge-reduce-llm's routing intelligence with AgentHandler's
    supervision. Every LLM call is:
    1. Routed by edge-reduce (local vs cloud vs workflow)
    2. Supervised by AgentHandler (budget, timeout, audit, cost tracking)

    The returned callable matches AgentHandler's LLM interface:
    ``async def llm(prompt: str) -> str``

    This means it plugs directly into PipelineBuilder, SkillBuilder,
    ReflectionLoop, and any LLM-backed guardrail.

    Args:
        supervisor: AgentHandler Supervisor for enforcement.
        local_base_url: URL of the local model (Ollama, vLLM, etc).
        cloud_api_key: API key for the cloud model.
        cloud_model: Cloud model name (default: claude-sonnet-4-20250514).
        local_model: Local model name (default: from Ollama).
        policy_path: Path to routing_policy.yaml.
        tool_name: Name used in audit log for LLM calls.
        pricing: PricingTable for cost attribution.
    """

    def __init__(
        self,
        supervisor: Supervisor,
        local_base_url: str = "http://localhost:11434/v1",
        cloud_api_key: Optional[str] = None,
        cloud_model: str = "claude-sonnet-4-20250514",
        local_model: Optional[str] = None,
        policy_path: Optional[str] = None,
        tool_name: str = "llm_call",
        pricing: Optional[PricingTable] = None,
    ) -> None:
        self._supervisor = supervisor
        self._local_base_url = local_base_url
        self._cloud_api_key = cloud_api_key
        self._cloud_model = cloud_model
        self._local_model = local_model
        self._policy_path = policy_path
        self._tool_name = tool_name
        self._pricing = pricing
        self._engine: Any = None
        self._last_result: Optional[EdgeReduceResult] = None

    def _get_engine(self) -> Any:
        """Lazy-init the RouterEngine from edge-reduce."""
        if self._engine is not None:
            return self._engine

        try:
            from localreduce.policy.schema import RoutingPolicy
            from localreduce.router.engine import RouterEngine
            from localreduce.router.tiers import CloudTier, LocalTier
        except ImportError as e:
            raise ImportError(
                "edge-reduce-llm required: pip install local-reduce\n"
                "Or: pip install git+https://github.com/arunvenkatadri/edge-reduce-llm.git"
            ) from e

        if self._policy_path:
            import yaml

            with open(self._policy_path) as f:
                policy = RoutingPolicy(**yaml.safe_load(f))
        else:
            policy = self._default_policy(RoutingPolicy)

        local = LocalTier(
            base_url=self._local_base_url,
            model=self._local_model,
        )
        cloud = CloudTier(
            api_key=self._cloud_api_key or "",
            model=self._cloud_model,
        )

        self._engine = RouterEngine(
            policy=policy,
            local=local,
            cloud=cloud,
        )
        return self._engine

    @staticmethod
    def _default_policy(RoutingPolicy: Any) -> Any:
        """Default: cloud-only. Configure a policy file to enable local routing."""
        return RoutingPolicy(
            rules=[
                {"name": "default", "match": {}, "route": "cloud"},
            ],
        )

    @classmethod
    def from_engine(
        cls,
        engine: Any,
        supervisor: Supervisor,
        tool_name: str = "llm_call",
    ) -> "EdgeReduceLLM":
        """Create from a pre-configured RouterEngine."""
        instance = cls(supervisor=supervisor, tool_name=tool_name)
        instance._engine = engine
        return instance

    async def __call__(self, prompt: str) -> str:
        """Route and execute an LLM call with supervision.

        This is the main interface — matches the ``async def llm(prompt) -> str``
        signature used throughout AgentHandler.
        """
        engine = self._get_engine()

        async def _routed_call(prompt: str = "") -> Dict[str, Any]:
            messages = [{"role": "user", "content": prompt}]
            result = await engine.handle(messages)

            self._last_result = EdgeReduceResult(
                text=result.text,
                route=result.route,
                model=result.model,
                confidence=result.confidence,
                tokens_local=result.tokens_local,
                tokens_cloud=result.tokens_cloud,
                cost_usd=result.cost_usd,
                latency_local_ms=result.latency_local_ms,
                latency_cloud_ms=result.latency_cloud_ms,
                rule_name=result.rule_name,
            )

            return {
                "result": result.text,
                "input_tokens": result.tokens_local + result.tokens_cloud,
                "output_tokens": 0,
                "model": result.model,
                "route": result.route,
                "confidence": result.confidence,
                "cost_usd": result.cost_usd,
            }

        supervised = await self._supervisor.call(self._tool_name, _routed_call, prompt=prompt)

        if supervised.succeeded:
            output = supervised.output
            if isinstance(output, dict):
                return str(output.get("result", ""))
            return str(output)

        error_msg = supervised.error.user_message() if supervised.error else "LLM call failed"
        raise RuntimeError(f"[AgentHandler] {error_msg}")

    @property
    def last_result(self) -> Optional[EdgeReduceResult]:
        """Get routing details from the most recent call."""
        return self._last_result

    async def call_with_details(self, prompt: str) -> EdgeReduceResult:
        """Call the LLM and return full routing details instead of just text."""
        await self(prompt)
        if self._last_result is None:
            raise RuntimeError("No result available")
        return self._last_result


def make_supervised_llm(
    supervisor: Supervisor,
    local_base_url: str = "http://localhost:11434/v1",
    cloud_api_key: Optional[str] = None,
    cloud_model: str = "claude-sonnet-4-20250514",
    policy_path: Optional[str] = None,
) -> Callable[[str], Coroutine[Any, Any, str]]:
    """Create a supervised hybrid LLM callable.

    Returns an ``async def llm(prompt: str) -> str`` that routes between
    local and cloud models via edge-reduce, with AgentHandler supervision.

    Usage:
        llm = make_supervised_llm(supervisor, cloud_api_key="sk-ant-...")
        pipeline = await PipelineBuilder(llm, tools, mgr).create("...")
    """
    return EdgeReduceLLM(
        supervisor=supervisor,
        local_base_url=local_base_url,
        cloud_api_key=cloud_api_key,
        cloud_model=cloud_model,
        policy_path=policy_path,
    )
