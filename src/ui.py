from typing import List
from langchain.chains.base import Chain
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
    r_print(f"[bold green]AI:[/bold green]", txt)
    r_print()


def chat_loop(chain: Chain, routes: List[Route]):
    print_ai("Welcome! I'm hear to answer questions or just chat. I have access to the following:")
    t = Table()
    t.add_column("Name")
    t.add_column("Description")
    [t.add_row(r.name, r.description) for r in routes]
    r_print(t)

    while True:
        query = prompt()
        with spinner("generating response"):
            result = chain(query)
        print_ai(result.get("text", result))
    
    