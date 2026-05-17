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


def chat(messages, client, model, system=None, temperature=1.0):
    params = {
            "model": model,
            "max_tokens": 1000,
            "messages": messages,
            "temperature": temperature,
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

    add_user_message(messages, "Generate a one sentence movie idea")

    answer = chat(messages, client, model, temperature=1.0)
    console.print(Markdown(answer))

if __name__ == "__main__":
    main()
