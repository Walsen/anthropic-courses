import os
import json
from anthropic import Anthropic
from rich.console import Console
from rich.json import JSON


def add_user_message(message, text):
    user_message = {"role": "user", "content": text}
    message.append(user_message)


# --- OLD WAY: assistant prefill + stop sequences (Claude 3.x only) ---
# Prefilling appends a partial assistant message so the model continues from that prefix.
# stop_sequences halts generation at the closing marker, yielding clean JSON.
# Removed in Claude 4.x (Sonnet 4.5+, Haiku 4.5+) — results in a 400 error.
#
# def add_assistant_message(message, text):
#     assistant_message = {"role": "assistant", "content": text}
#     message.append(assistant_message)


def generate_dataset(messages, client, model):
    prompt = """
    Generate an evaluation dataset for a prompt evaluation. The dataset will be used to evaluate prompts that generate Python, JSON, or Regex specifically for AWS-related tasks. Generate an array of JSON objects, each representing task that requires Python, JSON, or a Regex to complete.

    Example output:
    [
        {
            "task": "Description of task",
        },
        ...additional
    ]

    * Focus on tasks that can be solved by writing a single Python function, a single JSON object, or a single regex
    * Focus on tasks that do not require writing much code

    Please generate 3 objects.
    """

    # --- OLD WAY: prefill + stop sequence ---
    # add_user_message(messages, prompt)
    # add_assistant_message(messages, "```json")
    # text = chat(messages, client, model, stop_sequences=["```"])
    # return json.loads(text)

    # --- NEW WAY: system prompt instruction (Claude 4.x compatible) ---
    add_user_message(messages, prompt)
    system = "You are a JSON-only assistant. Always respond with a raw, valid JSON array and nothing else. No markdown, no explanation, no code fences."
    text = chat(messages, client, model, system=system)

    # Strip markdown code fences if the model includes them despite instructions
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1]  # remove opening ```json line
    if clean.endswith("```"):
        clean = clean.rsplit("```", 1)[0]  # remove closing ```

    return json.loads(clean.strip())


def run_prompt(test_case, client, model):
    """Merges the prompt and test case input, then returns the result"""
    prompt = f"""
Please solve the following task:

{test_case["task"]}
"""

    messages = []
    add_user_message(messages, prompt)
    output = chat(messages, client, model)
    return output


def run_test_case(test_case, client, model):
    """Calls run_prompt, then grades the result"""
    output = run_prompt(test_case, client, model)

    # TODO - Grading
    score = 10

    return {
        "output": output,
        "test_case": test_case,
        "score": score
    }


def run_eval(dataset, client, model):
    """Runs all test cases in the dataset and returns a list of results"""
    results = []
    for test_case in dataset:
        result = run_test_case(test_case, client, model)
        results.append(result)
    return results


def chat(messages, client, model, system=None, temperature=0.0, stop_sequences=[]):
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
    client = Anthropic()
    model = "claude-haiku-4-5-20251001"
    console = Console()

    messages = []

    dataset = generate_dataset(messages, client, model)
    console.print(JSON(json.dumps(dataset)))

    with open('dataset.json', 'w') as f:
        json.dump(dataset, f, indent=2)

    with open("dataset.json", "r") as f:
        dataset = json.load(f)

    results = run_eval(dataset, client, model)

    console.print(JSON(json.dumps(results)))


if __name__ == "__main__":
    main()
