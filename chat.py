import ollama

MODEL = "qwen2.5:3b"   # use "qwen2.5:1.5b" if responses are slow
SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and concisely."


def new_conversation():
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def main():
    messages = new_conversation()
    print(f"Chatting with {MODEL}. Type 'quit' to exit or 'reset' to clear memory.\n")

    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        if user_text.lower() == "quit":
            break
        if user_text.lower() == "reset":
            messages = new_conversation()
            print("(memory cleared)\n")
            continue

        messages.append({"role": "user", "content": user_text})
        try:
            response = ollama.chat(model=MODEL, messages=messages)
        except Exception as e:
            print(f"\nCould not reach the model: {e}")
            print("Make sure Ollama is open and you ran 'ollama pull " + MODEL + "'.\n")
            messages.pop()   # drop the message that failed
            continue

        reply = response["message"]["content"]
        messages.append({"role": "assistant", "content": reply})
        print(f"\nAI: {reply}\n")


if __name__ == "__main__":
    main()
