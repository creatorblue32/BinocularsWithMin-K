#IMPORTS
!pip install accelerate
!pip install datasets
!pip install tqdm

from typing import Union
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import accelerate
from google.colab import drive
import pandas as pd




#SETUPS
ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)

# selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527  # optimized for low-fpr [chosen at 0.01%]

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1

torch.set_grad_enabled(False)

#HELPER FUNCTIONS
def perplexity(encoding: transformers.BatchEncoding,
               logits: torch.Tensor,
               median: bool = False,
               temperature: float = 1.0):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    if median:
        ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                  masked_fill(~shifted_attention_mask.bool(), float("nan")))
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)

    else:
        ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
               shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.to("cpu").float().numpy()

    return ppl

def entropy(p_logits: torch.Tensor,
            q_logits: torch.Tensor,
            encoding: transformers.BatchEncoding,
            pad_token_id: int,
            median: bool = False,
            sample_p: bool = False,
            temperature: float = 1.0):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())

    return agg_ce

def assert_tokenizer_consistency(model_id_1, model_id_2):
    identical_tokenizers = (
            AutoTokenizer.from_pretrained(model_id_1).vocab
            == AutoTokenizer.from_pretrained(model_id_2).vocab
    )
    if not identical_tokenizers:
        raise ValueError(f"Tokenizers are not identical for {model_id_1} and {model_id_2}.")

class Binoculars(object):
    def __init__(self,
                 observer_name_or_path: str = "tiiuae/falcon-7b",
                 performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
                 use_bfloat16: bool = True,
                 max_token_observed: int = 512,
                 mode: str = "low-fpr",
                 cache_dir: str = None,
                 ) -> None:
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        self.change_mode(mode)

        # Use the cache directory if provided
        self.observer_model = AutoModelForCausalLM.from_pretrained(observer_name_or_path,
                                                                   cache_dir=cache_dir,
                                                                   device_map={"": DEVICE_1},
                                                                   trust_remote_code=True,
                                                                   torch_dtype=torch.bfloat16 if use_bfloat16
                                                                   else torch.float32,
                                                                   )
        self.performer_model = AutoModelForCausalLM.from_pretrained(performer_name_or_path,
                                                                    cache_dir=cache_dir,
                                                                    device_map={"": DEVICE_2},
                                                                    trust_remote_code=True,
                                                                    torch_dtype=torch.bfloat16 if use_bfloat16
                                                                    else torch.float32,
                                                                    )
        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def change_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = BINOCULARS_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE_2)).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                        encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()
        return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores

    def predict(self, input_text: Union[list[str], str]) -> Union[list[str], str]:
        binoculars_scores = np.array(self.compute_score(input_text))
        pred = np.where(binoculars_scores < self.threshold,
                        "Most likely AI-generated",
                        "Most likely human-generated"
                        ).tolist()
        return pred
    def __init__(self,
                 observer_name_or_path: str = "tiiuae/falcon-7b",
                 performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
                 use_bfloat16: bool = True,
                 max_token_observed: int = 512,
                 mode: str = "low-fpr",
                 ) -> None:
        print("NOOOO ABORT")
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        self.change_mode(mode)
        self.observer_model = AutoModelForCausalLM.from_pretrained(observer_name_or_path,
                                                                   device_map={"": DEVICE_1},
                                                                   trust_remote_code=True,
                                                                   torch_dtype=torch.bfloat16 if use_bfloat16
                                                                   else torch.float32,
                                                                   )
        self.performer_model = AutoModelForCausalLM.from_pretrained(performer_name_or_path,
                                                                    device_map={"": DEVICE_2},
                                                                    trust_remote_code=True,
                                                                    torch_dtype=torch.bfloat16 if use_bfloat16
                                                                    else torch.float32,
                                                                    )
        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def change_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = BINOCULARS_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE_2)).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_score(self, input_text):
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)

        # Compute perplexity
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                        encodings.to(DEVICE_1), self.tokenizer.pad_token_id)

        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()

        # Now compute min-k probabilities
        # Assuming you want to compute this for the performer model logits
        min_k_probs = self.min_k_probabilities(performer_logits, encodings)

        return binoculars_scores, min_k_probs

    def min_k_probabilities(self, logits, encoding, ratios=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        results = {}
        for ratio in ratios:
            k_length = int(len(encoding.input_ids[0]) * ratio)
            if k_length == 0:
                continue
            actual_token_probs = probabilities[0, :len(encoding.input_ids[0]), :].gather(1, encoding.input_ids.unsqueeze(-1)).squeeze(-1)
            sorted_probs = actual_token_probs.sort().values
            min_k_probs = sorted_probs[:k_length]
            results[f"Min_{int(ratio*100)}% Prob"] = -np.log(min_k_probs.mean()).item()
        return results


    def predict(self, input_text: Union[list[str], str]) -> Union[list[str], str]:
        binoculars_scores = np.array(self.compute_score(input_text))
        pred = np.where(binoculars_scores < self.threshold,
                        "Most likely AI-generated",
                        "Most likely human-generated"
                        ).tolist()
        return pred

def check_wikimia_dataset(dataset):
    seen_total = 0
    seen_mink = 0
    seen_quantity = 0
    unseen_total = 0
    unseen_mink = 0
    unseen_quantity = 0

    for example in dataset:
        binoculars_score, minkprobs = binoculars.compute_score(example['input'])
        label = example['label']
        if label == 1:
            seen_total += binoculars_score
            seen_mink += minkprobs
            seen_quantity += 1
        if label == 0:
            unseen_total += binoculars_score
            unseen_mink += minkprobs
            unseen_quantity += 1
    print("Seen Binoculars Average: " + (str(seen_total / seen_quantity) if seen_quantity != 0 else "") + " with quantity " + str(seen_quantity))
    print("Seen Mink Average: " + (str(seen_mink / seen_quantity) if seen_quantity != 0 else "") + " with quantity " + str(seen_quantity))
    print("Unseen Binoculars Average: " + (str(unseen_total / unseen_quantity) if unseen_quantity != 0 else "") + " with quantity " + str(unseen_quantity))
    print("Seen Mink Average: " + (str(unseen_mink / unseen_quantity) if unseen_quantity != 0 else "") + " with quantity " + str(unseen_quantity))

def merge_dataset(dataset):
    # Splitting the entries by label
    group_0 = [entry for entry in dataset if entry['label'] == 0]
    group_1 = [entry for entry in dataset if entry['label'] == 1]

    # Determine the minimum length to ensure balanced pairs
    min_length = min(len(group_0), len(group_1))

    # Create merged dataset
    merged_dataset = []

    # 0 is unseen, 1 is seen
    group_0_index = 0
    group_1_index = 0



    unseen_mode = True
    seen_first_mode = True

    while(group_0_index != len(group_0) and group_1_index != len(group_1)):
        if unseen_mode:
            string = group_0[group_0_index]['input']
            group_0_index += 1
            string += group_0[group_0_index]['input']
            group_0_index += 1
            unseen_mode = False
        else:
            if seen_first_mode:
                string = group_1[group_1_index]['input']
                group_1_index += 1
                string += group_0[group_0_index]['input']
                group_0_index += 1
                seen_first_mode = False
            else:
                string = group_0[group_0_index]['input']
                group_0_index += 1
                string += group_1[group_1_index]['input']
                group_1_index += 1
                seen_first_mode = True
            unseen_mode = True
        merged_dataset.append({'input': string, 'label': (0 if not unseen_mode else 1 )})

    return merged_dataset

import numpy as np

def compute_min_k_probabilities(all_prob, ratio):
    results = {}
    total_tokens = len(all_prob)
    k_length = int(total_tokens * ratio)  # Number of probabilities to consider
    if k_length == 0:
        raise ValueError("Ratio too small, resulting in zero tokens being considered.")
    topk_prob = np.sort(all_prob)[:k_length]  # Get the smallest k probabilities
    mean_topk_prob = -np.mean(topk_prob).item()  # Compute the negative mean of these probabilities
    return mean_topk_prob



# Processing Work:

# Retrieve Models from Cache
drive.mount('/content/drive')
cache_directory = "/content/drive/My Drive/transformers_cache/"

#Checking Vulnerable Texts
file_path = '/content/drive/My Drive/gdrive_datasets/vulnerable_texts.csv'
df = pd.read_csv(file_path)

for example in df:
    print("Text Name " + example['Text Name'])
    print("Text " + example['Text'])


# Create a Binoculars Object
binoculars = Binoculars(cache_dir=cache_directory)

for example in df:
    print("Text Name " + example['Text Name'])
    print(binoculars.compute_score(example['Text']))



LENGTH = 256
dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{LENGTH}")
check_wikimia_dataset(dataset)

