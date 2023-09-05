import torch

class LLMHandler:
    def __init__(self, llm_name):
        # self.llm =
        self.llm.eval()

if __name__ == '__main__':
    from transformers import AutoTokenizer
    import transformers
    import torch

    model_id = "meta-llama/Llama-2-13b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto')
    pipeline = transformers.pipeline(
        "text-generation",
        tokenizer=tokenizer,
        model=model,
        # torch_dtype=torch.float16,
        device_map="auto",
    )
    print('Created pipeline')
    sequences = pipeline(
        'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )
    print('Generated sequences')
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")



