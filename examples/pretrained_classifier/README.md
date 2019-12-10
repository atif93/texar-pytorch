# Classification using pre-trained models

This is a Texar implementation of a sentence classifier. Following models are supported: 

- Google's BERT model, which allows to load pre-trained model parameters downloaded
from the [official release](https://github.com/google-research/bert) and build/fine-tune arbitrary downstream
applications (This example showcases BERT for sentence classification).

- [OpenAI GPT-2 (Generative Pre-Trainning)](https://github.com/openai/gpt-2)
language model, which allows to load official pre-trained model parameters, generate samples, fine-tune the model,
and much more.

- [XLNet](https://github.com/zihangdai/xlnet). This example has reproduced the reported results on STS-B and IMDB on GPUs. As per
  [the official repository](https://github.com/zihangdai/xlnet#memory-issue-during-finetuning), computational resources
  (e.g., GPU memory) can affect the results.