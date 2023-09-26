from typing import Dict, List, Any, Optional, Sequence
from uuid import UUID
from langchain.chains.base import Chain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import AgentFinish, LLMResult
from langchain.schema.document import Document
from rich.table import Column
from rich.progress import Progress, SpinnerColumn, TextColumn, RenderableColumn
from rich.prompt import Prompt
from rich import print as r_print
from contextlib import contextmanager

from rich.table import Table

from src.logger import LoggingCallbackHandler
from src.models import Route

@contextmanager
def spinner(description: str, *args, **kwargs) -> Progress:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        *args,
        **kwargs
    ) as progress:
        progress.add_task(description, total=None)
        yield progress

def prompt(txt: str = "...."):
    return Prompt.ask(txt)

def print_ai(txt: str):
    r_print()
    r_print(f"[bold green]AI:[/bold green]", txt, end="")


def chat_loop(chain: Chain, routes: List[Route]):
    print_ai("Welcome! I'm hear to answer questions or just chat. I have access to the following:\n")
    t = Table()
    t.add_column("Name")
    t.add_column("Description")
    [t.add_row(r.name, r.description) for r in routes]
    r_print(t)
    print_ai("I'll do my best to use them based on what you type.")
    r_print(" If you want to force one of them, just use a bang operator.")
    r_print(f"Example: !{routes[0].name} will use {routes[0].name}")
    r_print()
    r_print()

    logging_callback = LoggingCallbackHandler()

    while True:
        query = prompt()
        with spinner("[bold green]AI:[/bold green]",
                     RenderableColumn(table_column=Column(overflow="fold")),
                     transient=False) as spin:
            chain(query, callbacks=[StreamOut(spin), logging_callback])
        r_print()

    
class StreamOut(StreamingStdOutCallbackHandler):
    
    def __init__(self, spin: Progress, *args, **kwargs):
        self._spin = spin
        super().__init__(*args, **kwargs)

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        r_print(f"[bright_black]route: {finish.log}[/]")
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self._spin.columns[-1].renderable += token
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        pass
        
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self._spin.columns = (RenderableColumn(" "), ) + self._spin.columns[1:]
        self._spin.refresh()
        self._spin.stop()
        
    def on_retriever_end(self, documents: Sequence[Document], *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        context = '\n\n----------\n\n'.join([d.page_content for d in documents])
        r_print(f"[bright_black]context: {context}[/]")