#IMPORTS
from typing import Union
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

#SETUPS
ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)

# selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527  # optimized for low-fpr [chosen at 0.01%]

# selected for Falcon-7B
MIN_K_RATIO = 0.05
MIN_K_THRESHOLD = 3.809450257187945e-08


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




def create_models(observer_name_or_path: str = "tiiuae/falcon-7b",
                  performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
                  use_bfloat16: bool = True,
                  cache_dir: str = None):
    observer_model = AutoModelForCausalLM.from_pretrained(
        observer_name_or_path,
        cache_dir=cache_dir,
        device_map={"": DEVICE_1},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
    )
    performer_model = AutoModelForCausalLM.from_pretrained(
        performer_name_or_path,
        cache_dir=cache_dir,
        device_map={"": DEVICE_2},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
    )
    return observer_model, performer_model


class Binoculars(object):
    def __init__(self, observer_model, performer_model, mode, max_token_observed):
        self.observer_model = observer_model
        self.performer_model = performer_model
        self.change_mode(mode)
        self.tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

        self.observer_model.eval()
        self.performer_model.eval()

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

    def compute_score_and_all_min_k(self, input_text):
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)

        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                        encodings.to(DEVICE_1), self.tokenizer.pad_token_id)

        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()

        # Now compute min-k probabilities
        min_k_probs = self.min_k_probabilities(observer_logits, encodings)

        return binoculars_scores, min_k_probs
    
    def compute_score_and_min_k(self, input_text, min_k_ratio, min_k_threshold):
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)

        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                        encodings.to(DEVICE_1), self.tokenizer.pad_token_id)

        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()

        # Now compute min-k probabilities
        min_k_prob = self.min_k_probability(observer_logits, encodings, min_k_ratio)
        memorized = (min_k_prob < min_k_threshold)
        memorized = memorized.tolist()


        return binoculars_scores, memorized

    
    def compute_score(self, input_text):
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)

        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                        encodings.to(DEVICE_1), self.tokenizer.pad_token_id)

        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()

        return binoculars_scores


    def min_k_probabilities(self, logits, encodings, k_ratios=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        results = {}
        input_ids = encodings['input_ids']

        for ratio in k_ratios:
            k_length = int(input_ids.size(1) * ratio)
            actual_token_probs = probabilities.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)
            sorted_probs, _ = actual_token_probs.sort(dim=1)
            min_k_probs = sorted_probs[:, :k_length]
            mean_prob = min_k_probs.mean(dim=1)
            results[ratio] = mean_prob
        return results

    def min_k_probability(self, logits, encodings, ratio):
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        results = {}
        input_ids = encodings['input_ids']
        k_length = int(input_ids.size(1) * ratio)
        actual_token_probs = probabilities.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)
        sorted_probs, _ = actual_token_probs.sort(dim=1)
        min_k_probs = sorted_probs[:, :k_length]
        mean_prob = min_k_probs.mean(dim=1)
        return mean_prob

    def predict(self, input_text: Union[list[str], str]) -> Union[list[str], str]:
        results_inside, memorized_inside = self.compute_score_and_min_k(input_text, MIN_K_RATIO, MIN_K_THRESHOLD)       
        binoculars_scores_inside = np.array(results_inside)
        predictions = []
        
        for score, mem in zip(binoculars_scores_inside, memorized_inside):
            if score < self.threshold:
                if mem:
                    predictions.append("Unable to determine authorship")
                else:
                    predictions.append("Most likely AI-generated")
            else:
                predictions.append("Most likely human-generated")

        return predictions
