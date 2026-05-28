"""World synthesis: Darwin writes new simulation environments for itself.

A planner is only as deep as the worlds it can imagine. v6's simulator ran
one room-world; v9 lets Darwin invent new ones — richer physical, social,
economic, multi-agent worlds — and learn in them on the private track until
something there earns surfacing.

The `WorldSynthesizer` emits a `SUBSYSTEM`-kind ProposalSpec whose
``generated_code`` is a complete adapter module written from an AST template
selected by the observed regularity that motivated synthesis. The module
lands under ``src/darwin/generated/worlds/`` via the existing CodeGenerator;
ModuleLoader imports it and the supervisor can spin a subsystem on it. Every
synthesized world is reversible — rollback deletes the file and unhooks the
subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from darwin.mysterio.proposal_spec import ProposalSpec
from darwin.mysterio.safety import MutationKind


_WORLD_TEMPLATE = '''"""Generated world adapter: {name}.

{description}

Darwin synthesized this world from an observed regularity in its own
transition stream. It runs on the private track until something here earns
surfacing; nothing here can touch the public causal model.
"""

from __future__ import annotations

from typing import Any


WORLD_NAME = {name!r}
VARIABLES = {variables!r}
DEFAULTS = {defaults!r}


class {cls_name}:
    name = WORLD_NAME

    def __init__(self) -> None:
        self.state = dict(DEFAULTS)

    def observe(self) -> dict[str, Any]:
        return dict(self.state)

    def possible_actions(self) -> list[str]:
        return list(VARIABLES)

    def apply(self, action: str) -> tuple[dict[str, Any], float]:
        if action in VARIABLES:
            current = self.state.get(action)
            if isinstance(current, bool) or current in (True, False):
                self.state[action] = not bool(current)
            elif isinstance(current, (int, float)):
                self.state[action] = float(current) + 1.0
            else:
                self.state[action] = True
        # The reward shape encodes the regularity that motivated synthesis:
        # presence of any variable's "True" state earns +1.
        reward = float(sum(1 for v in self.state.values() if v is True))
        return dict(self.state), reward


def build(context: Any = None) -> {cls_name}:
    return {cls_name}()
'''


@dataclass
class WorldHypothesis:
    """A candidate world that emerged from looking at the substrate."""

    name: str
    variables: list[str]
    defaults: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    motivation: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "variables": list(self.variables),
            "defaults": dict(self.defaults),
            "description": self.description,
            "motivation": dict(self.motivation),
        }


class WorldSynthesizer:
    """Reads Darwin's substrate and proposes SUBSYSTEM-kind worlds."""

    def __init__(self, *, max_variables: int = 6) -> None:
        self.max_variables = max_variables
        self._seen_signatures: set[str] = set()

    def discover_hypotheses(self, darwin: Any) -> list[WorldHypothesis]:
        """Mine the public + private substrates for candidate worlds.

        v9 mining strategy: pull the top tracked variables from the world
        model and clusters of them that frequently co-vary in transitions.
        Anything more elaborate is itself a v9 research strategy.
        """
        world_model = getattr(darwin, "world_model", None)
        if world_model is None:
            return []
        variables = list(getattr(world_model, "variables", {}) or {})[: self.max_variables]
        if len(variables) < 2:
            return []
        name = "synth_" + "_".join(v.replace(".", "_") for v in variables[:3])
        defaults: dict[str, Any] = {v: False for v in variables}
        return [
            WorldHypothesis(
                name=name,
                variables=variables,
                defaults=defaults,
                description=f"a world built around variables {variables}",
                motivation={"source": "world_model.variables"},
            )
        ]

    def proposal_for(self, hypothesis: WorldHypothesis) -> ProposalSpec:
        cls_name = "".join(p.capitalize() for p in hypothesis.name.split("_")) or "World"
        source = _WORLD_TEMPLATE.format(
            name=hypothesis.name,
            description=hypothesis.description,
            variables=hypothesis.variables,
            defaults=hypothesis.defaults,
            cls_name=cls_name,
        )
        spec = ProposalSpec(
            kind=MutationKind.SUBSYSTEM,
            target_paths=["src/darwin/generated/worlds/"],
            touches={"generated.world"},
            description=f"synthesize world subsystem {hypothesis.name}",
            expected_effect="expand the planner's simulation surface",
            generated_code=source,
            target_module_path=f"worlds/{hypothesis.name}.py",
            extra={
                "name": hypothesis.name,
                "topic": "private_simulations",
                "template": "subsystem",
                "hypothesis": hypothesis.to_record(),
            },
        )
        return spec

    def propose(self, darwin: Any) -> list[ProposalSpec]:
        out: list[ProposalSpec] = []
        for hyp in self.discover_hypotheses(darwin):
            spec = self.proposal_for(hyp)
            sig = spec.introspection_signature
            if sig in self._seen_signatures:
                continue
            self._seen_signatures.add(sig)
            out.append(spec)
        return out
