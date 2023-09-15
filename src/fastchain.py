import re
from pydantic import BaseModel
from langchain.chains.base import Chain
from langchain.chains.router import MultiRouteChain
from langchain.chains.router.embedding_router import EmbeddingRouterChain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.vectorstores import Chroma
from src.vectore_store import get_embeddings


from typing import Any, Dict, List, Optional


class Route(BaseModel):
    name: str
    description: str
    chain: Chain


class FastChain:

    def __init__(self) -> None:
        self.routes: List[Route] = []
        self._default_route: Route = None

    def add_route(self, route: Route, default_route=False):
        if not default_route:
            self.routes.append(route)
        else:
            self._default_route = route

    def add(self, name: str, description: str, chain: Chain, default_route=False):
        route = Route(
            name=name, description=description, chain=chain
        )
        self.add_route(route, default_route=default_route)

    def build_chain(self) -> Chain:
        router = BangEmbeddingRouterChain.from_names_and_descriptions(
            [(r.name, r.description,) for r in self.routes],
            Chroma,
            get_embeddings()
        )

        return MultiRouteChain(
            router_chain=router,
            destination_chains={r.name: r.chain for r in self.routes},
            default_chain=self._default_route.chain,
        )


class BangEmbeddingRouterChain(EmbeddingRouterChain):
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _input = ", ".join([inputs[k] for k in self.routing_keys])
        pattern = re.compile(r"^!(\w+)")
        match = pattern.match(_input)
        if match:
            word = match.group(1)
            _input = _input[match.end()+1:]
            return {"next_inputs": _input, "destination": word}
        return super()._call(inputs, run_manager)