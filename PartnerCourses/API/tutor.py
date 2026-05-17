import os
from anthropic import Anthropic
from rich.console import Console
from rich.markdown import Markdown


def add_user_message(message, text):
    user_message = {"role": "user", "content": text}
    message.append(user_message)


def add_assistant_message(message, text):
    assistant_message = {"role": "assistant", "content": text}
    message.append(assistant_message)


def chat(messages, client, model, system=None):
    params = {
            "model": model,
            "max_tokens": 1000,
            "messages": messages,
    }

    if system:
        params["system"] = system

    message = client.messages.create(**params)

    return message.content[0].text


def main():
    api = os.environ.get("ANTHROPIC_API_KEY")

    client = Anthropic()
    model = "claude-sonnet-4-6"
    console = Console()

    messages = []

    add_user_message(messages, "How do I solve 5x+2=3 for x?")

    # Without system prompt
    answer = chat(messages, client, model)
    console.print(Markdown(answer))
    input(">")

    # With system prompt
    system = '''
    You are a patient math tutor.
    Do not directly answer a student's questions.
    Guide them to a solution step by step.
    '''
    answer = chat(messages, client, model, system=system)
    console.print(Markdown(answer))

if __name__ == "__main__":
    main()
