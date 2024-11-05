from transformers import pipeline

model = pipeline(model="declare-lab/flan-alpaca-gpt4-xl")

def generate_prompt(keyword):
    # Input preparation
    prompt = f"Find similiar keywords to {keyword} for google search and write them comma separated."

    return model(prompt, max_length=25, do_sample=True)

# Examples
class_name = "plastic bag underwater litter"
prompt = generate_prompt(class_name)
print(f"Prompt generated '{class_name}': {prompt}")