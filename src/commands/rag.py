import typer
from src.vectore_store import load_or_create
from src.vectore_store import destroy as destroy_store
from src.models import RAG, AppSettings

app = typer.Typer()

@app.command(name="create")
def create(name: str, description: str, directory: str, persist_directory: str):
    settings = AppSettings()
    persist_dir = f"{persist_directory}/{settings.vector_storage_subdir}"
    load_or_create(data_dir=directory, persist_dir=persist_dir)
    rag = RAG(name=name, description=description, directory=directory, persist_directory=persist_dir)
    settings.rags.append(rag)
    settings.save()


@app.command(name="refresh")
def refresh(name: str):
    settings = AppSettings()
    for r in settings.rags:
        if r.name == name:
            load_or_create(data_dir=r.directory, persist_dir=r.persist_directory, refresh=True)


@app.command(name="destroy")
def destroy(name: str):
    settings = AppSettings()
    for r in settings.rags:
        if r.name == name:
            destroy_store(r.persist_directory)
    settings.rags = [r for r in settings.rags if r.name != name]
    settings.save()