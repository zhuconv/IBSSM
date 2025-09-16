import random
# First install textattack, tensorflow, and tensorflow_hub
# !pip install textattack tensorflow tensorflow_hub

import promptbench as pb
from promptbench.models import LLMModel
from promptbench.prompt_attack import Attack

# create model

# model_t5 = LLMModel(model='./out/tsz512x4k_20B_BIBS2/')
model_t5 = LLMModel(model='Mamba2', device='cuda',dtype='bfloat16')

# create dataset
dataset = pb.DatasetLoader.load_dataset("wnli")

# try part of the dataset
# dataset = dataset[:10]

# create prompt
prompt = "As a sentiment classifier, determine whether the following text is 'positive' or 'negative'. Please classify: \nQuestion: {content}\nAnswer:"

# define the projection function required by the output process
def proj_func(pred):
    mapping = {
        "positive": 1,
        "negative": 0
    }
    return mapping.get(pred, -1)

inv_mapping = {
    1: "positive",
    0: "negative"
}

# define the evaluation function required by the attack
# if the prompt does not require any dataset, for example, "write a poem", you still need to include the dataset parameter
indices = set(random.sample(range(len(dataset)), 5))

def eval_func(prompt, dataset, model):
    preds = []
    labels = []

    for i, d in enumerate(dataset):
        input_text = pb.InputProcess.basic_format(prompt, d)
        raw_output = model(input_text)
        if i in indices:
            print(f"Output: {raw_output}; Label: {inv_mapping[d['label']]}\n")
        

        output = pb.OutputProcess.cls(raw_output, proj_func)
        preds.append(output)

        labels.append(d["label"])
    
    return pb.Eval.compute_cls_accuracy(preds, labels)
    
# define the unmodifiable words in the prompt
# for example, the labels "positive" and "negative" are unmodifiable, and "content" is modifiable because it is a placeholder
# if your labels are enclosed with '', you need to add \' to the unmodifiable words (due to one feature of textattack)
unmodifiable_words = ["positive\'", "negative\'", "content"]

# print all supported attacks
print(Attack.attack_list())

# create attack, specify the model, dataset, prompt, evaluation function, and unmodifiable words
# verbose=True means that the attack will print the intermediate results
attack = Attack(model_t5, "stresstest", dataset, prompt, eval_func, unmodifiable_words, verbose=True)

# print attack result
print(attack.attack())

