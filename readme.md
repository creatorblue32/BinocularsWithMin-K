# Detecting memorized false positives in Binoculars text classification


<p align="center">
  <img src="assets/binocularswmink.png" width="300" height="300" alt="Binoculars with Mink"> <br> <em>Binoculars with Min-K%</em>
</p>


## Introduction and Problem

In a 2024 paper, a team of researchers proposed Binoculars, a zero-shot detection method for machine-generated text. They demonstrated good quantitative results by combining a perplexity with a metric called "cross-perplexity", which was introduced to combat high perplexity introduced by complex human prompts. 

This project (Binoculars with Min-K%) was created to deal with the a key weakness the researchers encountered in testing their solution: memorized false positives. In their discussion, they note that Binoculars performs poorly when the text is common or famous, and therefore has been "memorized" by the model. This is because these texts will have inherent low perplexity regardless of the actual contents. As a result, the researchers note that texts like the U.S. Constitution and Martin Luther King Jr.'s "I have a dream" speech are deemed by binoculars as AI-Generated.  

## Solution

There is a growing body of research aimed at detecting memorized pre-training data in the output logits of LLMs. The work we implement here is a 2023 paper that focuses on the problem. The researchers introduce a dataset to benchmark this task as well as propose and test a solution called Min-K Probability. 

In this project, we've determined the best Min-K probability ratio and threshold for the Falcon 7b model (used as the default observer in the original Binoculars implementation). Then, we introduce a stage in binoculars to check for memorized text using this threshold. If the original model deems a text "AI-Generated", it will trigger a Min-5% Probability check. If this check comes back below the threshold, the system will decline to classify the text. 

## Works Cited:

**Title:** Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text  
**Authors:** Abhimanyu Hans, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi, Aniruddha Saha, Micah Goldblum, Jonas Geiping, Tom Goldstein  
**Year:** 2024  
**eprint:** [2401.12070](https://arxiv.org/abs/2401.12070)  
**Archive Prefix:** arXiv  
**Primary Class:** cs.CL  

____

**Title:** Detecting Pretraining Data from Large Language Models  
**Authors:** Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, Luke Zettlemoyer  
**Year:** 2023  
**eprint:** [2310.16789](https://arxiv.org/abs/2310.16789)  
**Archive Prefix:** arXiv  
**Primary Class:** cs.CL  