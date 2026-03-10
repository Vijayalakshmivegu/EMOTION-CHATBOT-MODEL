from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Load trained model
MODEL_PATH = "emotion_chatbot_model"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

tokenizer.pad_token = tokenizer.eos_token

print("Model loaded successfully")


# Emotion helper responses
emotion_responses = {
    "sad": "I'm really sorry you're feeling sad. Do you want to talk about it?",
    "lonely": "Feeling lonely can be hard. I'm here to listen.",
    "happy": "That's great to hear! What made you happy today?",
    "angry": "It sounds like something upset you. Want to share what happened?",
    "tired": "You sound tired. Maybe taking a small break could help.",
    "depressed": "I'm really sorry you're feeling this way. You’re not alone."
}

# Safety words
critical_words = [
    "kill myself",
    "suicide",
    "want to die",
    "hurt myself",
    "hate myself"
]


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Chat API
@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()
    user_input = data["message"].strip().lower()

    if user_input == "":
        return jsonify({"response": "Please type something."})

    # Safety filter
    for word in critical_words:
        if word in user_input:
            return jsonify({
                "response": "I'm really sorry you're feeling this way. You deserve support. Please consider reaching out to a trusted friend, family member, or professional."
            })

    # Emotion detection shortcut
    for emotion in emotion_responses:
        if emotion in user_input:
            return jsonify({
                "response": emotion_responses[emotion]
            })

    # AI model response
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
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    response = decoded.replace(prompt, "").strip()

    if response == "":
        response = "I'm here to listen. Tell me more about how you're feeling."

    return jsonify({"response": response})


# Run server
if __name__ == "__main__":
    app.run(debug=True)