# MatSciBERT
A Materials Domain Language Model for Text Mining and Information Extraction

## Using MatSciBERT

### Importing libraries
```
import torch
from normalize_text import normalize
from transformers import AutoModel, AutoTokenizer
```

### Loading pretrained model and tokenizer
```
tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert')
model = AutoModel.from_pretrained('m3rg-iitd/matscibert')
```

### Tokenizing sentences
```
sentences = ['SiO2 is a network former.']
norm_sents = [normalize(s) for s in sentences]
tokenized_sents = tokenizer(norm_sents)
tokenized_sents = {k: torch.Tensor(v).long() for k, v in tokenized_sents.items()}
```

### Obtaining BERT embeddings
```
with torch.no_grad():
    last_hidden_state = model(**tokenized_sents)[0]
```
## Citing

If you use MatSciBERT in your research, please cite [MatSciBERT: A Materials Domain Language Model for Text Mining and Information Extraction](https://arxiv.org/abs/2109.15290)
```
@article{gupta_matscibert_2021,
  title = {{{MatSciBERT}}: A {{Materials Domain Language Model}} for {{Text Mining}} and {{Information Extraction}}},
  shorttitle = {{{MatSciBERT}}},
  author = {Gupta, Tanishq and Zaki, Mohd and Krishnan, N. M. Anoop and Mausam},
  year = {2021},
  month = sep,
  journal = {arXiv:2109.15290 [cond-mat]},
  eprint = {2109.15290},
  eprinttype = {arxiv},
  primaryclass = {cond-mat},
  archiveprefix = {arXiv},
  keywords = {Computer Science - Computation and Language,Condensed Matter - Materials Science}}
}
```