import os
from anthropic import Anthropic


def add_user_message(message, text):
    user_message = {"role": "user", "content": text}
    message.append(user_message)


def add_assistant_message(message, text):
    assistant_message = {"role": "assistant", "content": text}
    message.append(assistant_message)


def chat(messages, client, model):
    message = client.messages.create(
            model=model,
            max_tokens=1000,
            messages=messages,
    )
    return message.content[0].text


def main():
    api = os.environ.get("ANTHROPIC_API_KEY")

    client = Anthropic()
    model = "claude-sonnet-4-6"
    
    messages = []

    while True:
        user_input = input("> ")
        print(">")

        add_user_message(messages, user_input)
        answer = chat(messages, client, model)
        add_assistant_message(messages, answer)
        print("---")
        print(answer)
        print("---")


if __name__ == "__main__":
    main()
