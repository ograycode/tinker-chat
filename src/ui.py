from typing import Dict, List, Any
from langchain.chains.base import Chain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Prompt
from rich import print as r_print
from contextlib import contextmanager

from rich.table import Table

from src.fastchain import FastChain, Route

@contextmanager
def spinner(description: str, **kwargs):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
        **kwargs
    ) as progress:
        progress.add_task(description, total=None)
        yield progress

def prompt(txt: str = "[bold]>>>[/bold]"):
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

    while True:
        query = prompt()
        chain(query, callbacks=[StreamOut()])
        r_print()
        r_print()

    
class StreamOut(StreamingStdOutCallbackHandler):
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        r_print(token, end="")
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        print_ai("")