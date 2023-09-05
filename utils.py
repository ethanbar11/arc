import pyperclip
# Import the os and random modules
import os
import random
import json
import torch
import seaborn as sns
import matplotlib.pyplot as plt

numbers_to_letters = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', }


def convert_matrix_to_string(matrix):
    # Convert the matrix to a string s.t only thing seperating the numbers is a space,
    # and each row is on a new line
    matrix_as_string = ""
    for row in matrix:
        for number in row:
            matrix_as_string += str(number) + " "
        matrix_as_string += "\n"
    return matrix_as_string


def convert_matrix_to_rgb(matrix):
    # Plot matrix as the following colors: 0: black, 1: blue, 2: red 3: green
    # 4: yellow 5: grey 6: fuchsia 7: orange 8: teal 9: brown
    colors = {
        0: (0, 0, 0),
        1: (0, 0, 255),
        2: (255, 0, 0),
        3: (0, 255, 0),
        4: (255, 255, 0),
        5: (128, 128, 128),
        6: (255, 0, 255),
        7: (255, 165, 0),
        8: (0, 128, 128),
        9: (165, 42, 42)
    }
    # Create a new matrix with the same shape as the input matrix
    new_matrix = torch.zeros((matrix.shape[0], matrix.shape[1], 3))
    # Iterate over the matrix and replace each number with the corresponding color
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            new_matrix[i, j] = torch.tensor(colors[matrix[i, j].item()]) / 255.0
    # Return the new matrix
    return new_matrix


# Define a function that takes a directory path as an argument
def get_random_file_path(dir_path):
    # Get a list of all the files in the directory
    files = os.listdir(dir_path)
    # If the list is not empty, choose a random file and join it with the directory path
    if files:
        file = random.choice(files)
        file_path = os.path.join(dir_path, file)
        # Return the file path
        return file_path
    # If the list is empty, return None
    else:
        return None


class Task:
    def __init__(self, task_as_json, name=None):
        self.json = task_as_json
        self.train = task_as_json['train']
        self.train_size = len(self.train)
        self.test = task_as_json['test']
        self.name = name

    def plot_train_examples(self):
        for example in self.train:
            inp = example['input']
            out = example['output']
            # Convert each matrix to torch and plot them using seaborn as heatmap
            inp = torch.tensor(inp)
            out = torch.tensor(out)
            fig, (ax1, ax2) = plt.subplots(1, 2)
            inp_as_rgb = convert_matrix_to_rgb(inp)
            out_as_rgb = convert_matrix_to_rgb(out)
            # Plot inp_as_rgb as image
            ax1.imshow(inp_as_rgb)
            ax2.imshow(out_as_rgb)
            plt.show()

    def to_prompt(self):
        with open('prompt_creators/prompt_step_by_step_without_examples.txt', 'r') as f:
            start_prompt = f.read()
        start_prompt = start_prompt.format(len(self.train))
        training_prompt = "\n"
        for idx,example in enumerate(self.train):
            inp = convert_matrix_to_string(example['input'])
            out = convert_matrix_to_string(example['output'])
            training_prompt += numbers_to_letters[idx] + ".\n"
            training_prompt += f"input:\n{inp}\noutput:\n{out}\n"
        test_prompt = f"Test:\ninput:\n{convert_matrix_to_string(self.test[0]['input'])}"
        return start_prompt + training_prompt + test_prompt

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def compare_to_output(self):
        output = input()
        gpt_out = torch.Tensor(eval(output))
        # Plot the input matrix and the correct and predicted output matrices
        # With legends and titles
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        inp_as_rgb = convert_matrix_to_rgb(torch.tensor(self.test[0]['input']))
        out_as_rgb = convert_matrix_to_rgb(torch.tensor(self.test[0]['output']))
        gpt_out_as_rgb = convert_matrix_to_rgb(torch.tensor(gpt_out))
        ax1.imshow(inp_as_rgb)
        ax1.set_title('Input')
        ax2.imshow(out_as_rgb)
        ax2.set_title('Correct Output')
        ax3.imshow(gpt_out_as_rgb)
        ax3.set_title('Predicted Output')
        plt.show()


class ArcTasks:
    def __init__(self, path):
        self.path = path
        self.training = []
        self.evaluation = []
        self.load_tasks()

    def load_tasks(self):
        for file in os.listdir(os.path.join(self.path, 'training')):
            # Read task as json
            with open(os.path.join(self.path, 'training', file), 'r') as f:
                task_as_json = json.load(f)
            self.training.append(Task(task_as_json, name=file))
        for file in os.listdir(os.path.join(self.path, 'evaluation')):
            with open(os.path.join(self.path, 'evaluation', file), 'r') as f:
                task_as_json = json.load(f)
            self.evaluation.append(Task(task_as_json, name=file))


def get_task(name=None, training=False):
    # get current directory
    cwd = os.getcwd()
    path = os.path.join(cwd, 'data', 'training') if training else os.path.join(cwd, 'data', 'evaluation')
    if name is None:
        # Get random file from the path directory
        file_path = get_random_file_path(path)
    else:
        file_path = os.path.join(path, name)
    # Read the file as json with the right library
    with open(file_path, 'r') as f:
        task_as_json = json.load(f)
    # Return the task
    return Task(task_as_json, name=name)


if __name__ == '__main__':
    # name = '0c786b71outp.json'
    #     # task = get_task(name=name, training=False)
    #     # print(task)
    #     # # task.plot_train_examples()
    #     # prompt = task.to_prompt()
    #     # pyperclip.copy(prompt)
    #     # print('prompt length:', len(prompt))
    #     # print(prompt)
    #     # task.compare_to_ut()

    path = os.path.join(os.getcwd(), 'data')
    arc = ArcTasks(path)
    tasks_lengths=[]
    for task in arc.evaluation:
        prompt = task.to_prompt()
        tasks_lengths.append((task,len(prompt)))
    # Sort them
    tasks_lengths.sort(key=lambda x: x[1], reverse=True)
    for task,length in tasks_lengths:
        print(task,length)
