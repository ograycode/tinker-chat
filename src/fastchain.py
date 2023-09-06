from pydantic import BaseModel
from langchain.chains.base import Chain
from langchain.chains.router import MultiRouteChain
from langchain.chains.router.embedding_router import EmbeddingRouterChain
from langchain.vectorstores import Chroma
from src.vectore_store import get_embeddings


from typing import List


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

    def create(self) -> Chain:
        router = EmbeddingRouterChain.from_names_and_descriptions(
            [(r.name, r.description,) for r in self.routes],
            Chroma,
            get_embeddings()
        )

        return MultiRouteChain(
            router_chain=router,
            destination_chains={r.name: r.chain for r in self.routes},
            default_chain=self._default_route.chain,
        )