from rich.console import Console
from rich.text import Text
import re

def print_colored_terminal(words, scores):
    """
    Prints the words to the console with colors based on the scores.
    Green is used for positive scores and red is used for negative scores.
    """
    
    console = Console()
    txt = Text()
    for w, s in zip(words, scores):
        style = "green" if s>=0 else "red"
        txt.append(w + ' ', style=style)
    console.print(txt)
    


def remove_thinking(text: str) -> str:
    """
    Remove all <think>...</think> tags and everything between them.
    
    Args:
        text: Input string potentially containing one or more <think>…</think> blocks.
        
    Returns:
        String with those blocks stripped out.
    """
    # DOTALL so `.` matches newlines
    pattern = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
    return pattern.sub("", text)


