numbers_to_letters = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j'}
letters_to_numbers = {v: k for k, v in numbers_to_letters.items()}
import re
numbers_to_text_numbers = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
                           5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
text_numbers_to_numbers = {v: k for k, v in numbers_to_text_numbers.items()}


class ZeroShotPromptConvertor:
    def __init__(self, prompt_start_location, using_numbers=False, delimiter=' '):
        if not delimiter:
            delimiter = ' '
        self.to_prompt_dict = numbers_to_text_numbers if using_numbers else numbers_to_letters
        self.from_prompt_dict = text_numbers_to_numbers if using_numbers else letters_to_numbers

        self.delimiter = delimiter
        with open(prompt_start_location, 'r') as f:
            self.prompt_start = f.read()

    def convert_task_to_prompt(self, train, test):
        start_prompt = self.prompt_start  # .format(len(self.train))
        training_prompt = "\n"
        for idx, example in enumerate(train):
            inp = self.convert_matrix_to_string(example['input'], self.delimiter)
            out = self.convert_matrix_to_string(example['output'], self.delimiter)
            training_prompt += numbers_to_letters[idx] + ".\n"
            training_prompt += f"input:\n{inp}\noutput:\n{out}\n"
        test_prompt = f"Test:\ninput:\n{self.convert_matrix_to_string(test[0]['input'], self.delimiter)} \nAnswer:"
        return start_prompt + training_prompt + test_prompt

    def convert_matrix_to_string(self, matrix, delimiter):
        # Convert the matrix to a string s.t only thing separating the numbers is a space,
        # and each row is on a new line
        matrix_as_string = "--\n"
        for row in matrix:
            for number in row:
                matrix_as_string += self.to_prompt_dict[number] + delimiter
            matrix_as_string += "\n"
        matrix_as_string += "--"
        return matrix_as_string

    def convert_text_to_mat(self, text):
        pattern = r"((?:[a-j1-9] )+[a-j1-9]\s*\n)+"
        matrices = []
        text_matrices = re.finditer(pattern, text)
        if text_matrices:
            for i, match in enumerate(text_matrices):
                mat_as_text = match.group()
                mat = []
                text_lines = mat_as_text.split('\n')
                for text_line in text_lines:
                    if text_line == '':
                        continue
                    line = []
                    for element in text_line.split(self.delimiter):
                        if element == '':
                            continue
                        line.append(self.from_prompt_dict[element])
                    mat.append(line)
                matrices.append(mat)
        else:
            print("No matrices found")
        print('Found {} matrices'.format(len(matrices)))

        return matrices

# if __name__ == '__main__':
#     txt = """
#
# --
# 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 2 0 0 0 0
# 0 0 0 0 0 2 2 0 0 0
# 0 8 8 0 0 2 2 0 0 0
# 0 8 8 0 0 0 2 0 0 0
# 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0
# --
#
# Please provide the transformation matrix for the given riddles.
#
# Note: The transformation matrix should be a square matrix of size 2x2 or 3x3, and it should be applied element-wise to the input matrix."""
#     start_path = 'prompt_creators/prompt_zero_shot.txt'
#     ZeroShotPromptConvertor(start_path,using_numbers=True).convert_text_mat(txt)
