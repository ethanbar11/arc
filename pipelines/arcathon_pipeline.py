from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


class ArcathonPipeline(torch.nn.Module):
    def __init__(self, prompt_convertor, model_id, max_return_length, temp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_convertor = prompt_convertor
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='cuda:0')
        self.model.eval()
        self.pipeline = pipeline(
            "text-generation",
            tokenizer=self.tokenizer,
            model=self.model,
            # torch_dtype=torch.float16,
            device_map="auto",
        )
        self.max_return_length = max_return_length
        self.temperature = temp

    def forward(self, train, test):
        prompt = self.prompt_convertor.convert_task_to_prompt(train, test)
        sequences = self.pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=self.max_return_length,
            temperature=self.temperature
        )
        out = sequences[0]['generated_text'][len(prompt):]
        print(f'Inserted prompt {prompt}')
        print(f'Length of output {len(sequences[0]["generated_text"]) - len(prompt)}')
        print('Output:')
        print(out)
        out_matrices = self.prompt_convertor.convert_text_to_mat(out)
        success = False
        for mat in out_matrices:
            if test[0]['output'] == mat:
                success = True
        return success, out_matrices, out
