import random
import argparse
# First install textattack
import promptbench as pb
from promptbench.models import LLMModel
from promptbench.prompt_attack import attack_config
# attack_config["goal_function"]['query_budget'] = 100  # set the query budget to 100



from promptbench.prompt_attack import Attack

def parse_args():
    parser = argparse.ArgumentParser(description="Run promptbench attacks with args for model and attack type")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Model path or identifier (e.g. 'mistralai/Mamba-Codestral-7B-v0.1')")
    parser.add_argument("-a", "--attack", type=str, required=True,
                        help="Attack name (e.g. textbugger, deepwordbug, textfooler, bertattack, stresstest)")
    parser.add_argument("-d", "--dataset", type=str, default="sst2",
                        help="Dataset name to load via promptbench (default: sst2)")
    parser.add_argument("--n_examples", type=int, default=100,
                        help="Take first N examples from dataset (default: 100)")
    parser.add_argument("--sample_k", type=int, default=5,
                        help="Number of indices to print example outputs for debugging (default: 5)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run model on (default: cuda)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="DType for model (default: bfloat16)")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature for the model (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose attack printing")
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)

    # create model
    print(f"Loading model: {args.model} on {args.device} with dtype {args.dtype}")
    model = LLMModel(model=args.model, device=args.device, dtype=args.dtype, temperature=args.temperature)

    # load dataset
    dataset = pb.DatasetLoader.load_dataset(args.dataset)
    # try part of the dataset
    if args.n_examples is not None:
        dataset = dataset[:args.n_examples]


    # create prompt
    promtp = """As a sentiment classifier, determine whether the following text is "positive" or "negative". Only output one word: either "positive" or "negative". No explanation. \nText: {content}\nAnswer:"""
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
    sample_k = min(args.sample_k, len(dataset))
    indices = set(random.sample(range(len(dataset)), sample_k))

    def eval_func(prompt, dataset, model):
        preds = []
        labels = []

        for i, d in enumerate(dataset):
            input_text = pb.InputProcess.basic_format(prompt, d)
            raw_output = model(input_text)
            raw_output = raw_output.replace("</s>", "").replace("<s>", "").strip()
            raw_output = raw_output[len(input_text):].strip()
            # if i in indices:
                # print(f"*** OUTPUT: {raw_output}; LABEL: {inv_mapping[d['label']]} ***")
            

            output = pb.OutputProcess.cls(raw_output, proj_func)
            preds.append(output)

            labels.append(d["label"])
        
        return pb.Eval.compute_cls_accuracy(preds, labels)
        
    # define the unmodifiable words in the prompt
    # for example, the labels "positive" and "negative" are unmodifiable, and "content" is modifiable because it is a placeholder
    # if your labels are enclosed with '', you need to add \' to the unmodifiable words (due to one feature of textattack)
    unmodifiable_words = ["positive\'", "negative\'", "content"]

    # print all supported attacks
    print("Supported Attacks:", Attack.attack_list())

    # create attack, specify the model, dataset, prompt, evaluation function, and unmodifiable words
    # verbose=True means that the attack will print the intermediate results
    # 'textfooler', 'bertattack', 'stresstest' are used here
    attack = Attack(model, args.attack, dataset, prompt, eval_func, unmodifiable_words, verbose=args.verbose)

    # print attack result
    result = attack.attack()
    print("Attack Result:", result)



if __name__ == "__main__":
    main()