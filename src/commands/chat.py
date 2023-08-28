from src.ai import create_ai
from src.ui import spinner, chat_loop

def main():

    with spinner("building..."):
        ai = create_ai()
        chain = ai.create()
    chat_loop(chain, ai.routes)


