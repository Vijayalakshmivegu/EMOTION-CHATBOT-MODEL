import json

with open("evaluation/evaluation_prompts.json") as f:
    prompts = json.load(f)

results = []

for item in prompts:
    prompt = item["prompt"]
    response = "Sample model response"

    results.append({
        "prompt": prompt,
        "response": response
    })

with open("evaluation/model_outputs.json", "w") as f:
    json.dump(results, f, indent=2)

print("Evaluation completed.")