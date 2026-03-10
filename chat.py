from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "emotion_chatbot_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

tokenizer.pad_token = tokenizer.eos_token

print("\nEmotion AI Chatbot Ready!")
print("Type 'quit' to exit.\n")

# basic emotional safety keywords
critical_words = [
    "hate myself",
    "kill myself",
    "suicide",
    "want to die",
    "hurt myself"
]

while True:

    user_input = input("You: ").strip().lower()

    if user_input == "":
        print("AI: Please type something.")
        continue

    if user_input == "quit":
        break

    # safety response
    if any(word in user_input for word in critical_words):
        print("AI: I'm really sorry you're feeling this way. You deserve support and care. Talking to a trusted friend, family member, or professional can really help.")
        continue

    prompt = f"User: {user_input}\nAI:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50,
        temperature=0.8,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    response = decoded.replace(prompt, "").strip()

    if response == "":
        response = "I'm here to listen. Tell me more about how you're feeling."

    print("AI:", response)