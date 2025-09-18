import llama_cpp
import json
import os
import re

# Load the Llama model
llm = llama_cpp.Llama(
    model_path=r"C:\Users\User\Desktop\Mint HRM\company_chatbot\model\llama-2-7b-chat.Q3_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=40,  # try 20 first; adjust up/down based on VRAM
    verbose=False,
)

# File paths for storing history
HISTORY_FILE = "chat_history.json"
FACTS_FILE = "facts.json"
MAX_HISTORY = 5


# Load persistent memory
def load_memory(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return []


# Save persistent memory
def save_memory(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f)


# Load stored data
conversation_history = load_memory(HISTORY_FILE)
important_facts = load_memory(FACTS_FILE)


def extract_location(user_input):
    """Extracts location after 'in' (e.g., 'in Dublin')."""
    match = re.search(r" in ([a-zA-Z\s]+)", user_input)
    if match:
        return match.group(1).strip()
    return None


def detect_fact(user_input):
    """Detects important facts to remember."""
    if "my name is" in user_input:
        return user_input
    if "i live in" in user_input:
        return user_input
    if "i work as" in user_input:
        return user_input
    return None


def process_task(user_input):
    """Process user tasks like checking time, remembering facts, etc."""
    global important_facts

    user_input = user_input.lower()
    location = extract_location(user_input)

    # Save important facts
    fact = detect_fact(user_input)
    if fact and fact not in important_facts:
        important_facts.append(fact)
        save_memory(FACTS_FILE, important_facts)
        return "Got it! I'll remember that."

    # Check stored facts if user asks about themselves
    if "what's my name" in user_input:
        for fact in important_facts:
            if "my name is" in fact:
                return fact.replace("my name is", "Your name is")
        return "I don't know your name yet. You can tell me by saying 'My name is [your name]'."

    return None


def chatbot():
    global conversation_history
    print("AI Assistant: How can I assist you today? (type 'exit' to quit)")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("AI Assistant: Goodbye!")
            save_memory(HISTORY_FILE, conversation_history)  # Save chat history
            break

        # First, check if it's a known task
        task_response = process_task(user_input)
        if task_response:
            print(f"AI Assistant: {task_response}")
            continue

        # Append user input to short-term memory
        conversation_history.append(f"\n You: {user_input}")
        if len(conversation_history) > MAX_HISTORY * 2:  # Limit history size
            conversation_history.pop(0)

        # Format prompt with facts + conversation history
        prompt = (
            "\n".join(important_facts)
            + "\n"
            + "\n".join(conversation_history)
            + "\nAI Assistant:"
        )

        # Generate response
        response = llm(prompt, max_tokens=200)
        assistant_reply = response["choices"][0]["text"].strip()

        # Append AI response to history
        conversation_history.append(f"AI Assistant: {assistant_reply}")

        print(f"\n AI Assistant: {assistant_reply} \n")


if __name__ == "__main__":
    chatbot()
