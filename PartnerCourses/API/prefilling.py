import os
import json
from anthropic import Anthropic
from rich.console import Console
from rich.json import JSON


def add_user_message(message, text):
    user_message = {"role": "user", "content": text}
    message.append(user_message)


def add_assistant_message(message, text):
    assistant_message = {"role": "assistant", "content": text}
    message.append(assistant_message)


def chat(messages, client, model, system=None, temperature=0.0, stop_sequences=None):
    params = {
            "model": model,
            "max_tokens": 1000,
            "messages": messages,
            "temperature": temperature,
    }

    if system:
        params["system"] = system

    if stop_sequences:
        params["stop_sequences"] = stop_sequences

    message = client.messages.create(**params)

    return message.content[0].text


def main():
    api = os.environ.get("ANTHROPIC_API_KEY")

    client = Anthropic()
    model = "claude-sonnet-4-6"
    console = Console()

    messages = []

    # --- OLD WAY: assistant prefill + stop sequences (Claude 3.x only) ---
    # Prefilling works by appending a partial assistant message before the API call.
    # The model continues from that prefix, and stop_sequences halts it at the closing marker.
    # This was removed in Claude 4.x (Sonnet 4.5+ / Sonnet 4.6+) — results in a 400 error.
    #
    # add_user_message(messages, "Generate a very short event bridge rule as json")
    # add_assistant_message(messages, "```json")  # prefill: forces model to start with ```json
    # text = chat(messages, client, model, stop_sequences=["```"])  # stop before closing ```
    # clean_json = json.loads(text.strip())
    # print(clean_json)

    # --- NEW WAY: system prompt instruction (Claude 4.x compatible) ---
    # Instead of prefilling, instruct the model via the system prompt to return raw JSON only.
    # The model respects this without needing a prefill prefix or stop sequences.
    system = "You are a JSON-only assistant. Always respond with raw, valid JSON and nothing else. No markdown, no explanation, no code fences."

    add_user_message(messages, "Generate a very short EventBridge rule as JSON.")

    text = chat(messages, client, model, system=system)

    # Clean up and parse the JSON
    clean_json = json.loads(text.strip())
    console.print(JSON(json.dumps(clean_json)))


if __name__ == "__main__":
    main()
