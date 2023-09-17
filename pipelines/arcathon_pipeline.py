import timeit

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import torch


class ArcathonPipeline(torch.nn.Module):
    def __init__(self, prompt_convertor, model_id, max_new_tokens, min_new_tokens, top_p, temperature, device,
                 tries_amount, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_convertor = prompt_convertor
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map=device,
                                                          trust_remote_code=True)
        self.generation_config = GenerationConfig(max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                                  top_p=top_p, temperature=temperature, do_sample=True)
        self.tries_amount = tries_amount
        self.model.eval()

    def forward(self, train, test):
        success = False
        saved_outputs = []
        for i in range(self.tries_amount):
            start = timeit.default_timer()
            prompt = self.prompt_convertor.convert_task_to_prompt(train, test)
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            text_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            to_find = '\n assistant\n'
            location = text_output.find(to_find)
            out = text_output[location + len(to_find):]
            saved_outputs.append(out)
            print(f'Inserted prompt {prompt}')
            print(f'Length of output {len(text_output) - len(prompt)}')
            print('Output:')
            print(out)
            real_out_mat_as_txt = self.prompt_convertor.convert_mat_to_text(test[0]['output'])
            print(f"Real output: {real_out_mat_as_txt}")
            if real_out_mat_as_txt in out:
                success = True
            end = timeit.default_timer()
            if success:
                print(f'Success! Generation took {end - start} seconds')
                break
            else:
                if i+1 == self.tries_amount:
                    print(f'Failed to generate correct output completly.')
                else:
                    print(f'Failed to generate correct output, try : {i + 1} / {self.tries_amount}. trying again')
                print(f'Generation took {end - start}s')
        return success, saved_outputs


if __name__ == '__main__':
    model_id = r'OpenAssistant/codellama-13b-oasst-sft-v10'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    device = 'cuda:2'
    model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map=device, trust_remote_code=True)
    input_text = """
<|im_start|>system
You are the boss of programming<|im_end|>
<|im_start|>user
Write me a function that sums 2 numbers in Python:<|im_end|>
<|im_start|>assistant
"""
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        outputs = model.generate(**inputs)

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print('woho')
