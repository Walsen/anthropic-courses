import os
from anthropic import Anthropic


def add_user_message(message, text):
    user_message = {"role": "user", "content": text}
    message.append(user_message)


def add_assistant_message(message, text):
    assistant_message = {"role": "assistant", "content": text}
    message.append(assistant_message)


def chat(messages, client, model, system=None, temperature=0.0):
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
    
    messages = []

    add_user_message(messages, "Write a 1 sentence description of a fake database")

    with client.messages.stream(
        model=model,
        max_tokens=1000,
        messages=messages
    ) as stream:
        for text in stream.text_stream:
            print(text, end="")
            # pass
    
    # Get the complete message for database storage
    # final_message = stream.get_final_message()


if __name__ == "__main__":
    main()
