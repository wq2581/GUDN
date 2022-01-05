import os
import re
import yaml
import torch
import spacy
import numpy as np


class NLP:
    def __init__(self, path):
        self.nlp = spacy.load(path, disable=['ner', 'parser', 'tagger'])
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def sent_tokenize(self, text):
        doc = self.nlp(text)
        sentences = [sent.string.strip() for sent in doc.sents]
        return sentences

    def word_tokenize(self, text, lower=False):  # create a tokenizer function
        if text is None:
            return text
        text = ' '.join(text.split())
        if lower:
            text = text.lower()
        toks = [tok.text for tok in self.nlp.tokenizer(text)]
        return ' '.join(toks)


class Accuracy:
    def __init__(self):
        super(Accuracy, self).__init__()
        self.total = 0
        self.acc1 = 0
        self.acc3 = 0
        self.acc5 = 0

    def calc(self, logits, labels):
        acc1, acc3, acc5, total = get_accuracy(logits, labels)
        self.total += total
        self.acc1 += acc1
        self.acc3 += acc3
        self.acc5 += acc5

    def get_accuracy(self, logits, labels):
        scores, indices = torch.topk(logits.detach().cpu(), k=10)

        acc1, acc3, acc5, total = 0, 0, 0, 0
        for index, label in enumerate(labels.cpu().numpy()):
            label = set(np.nonzero(label)[0])

            labels = indices[index, :5].numpy()

            acc1 += len(set([labels[0]]) & label)
            acc3 += len(set(labels[:3]) & label)
            acc5 += len(set(labels[:5]) & label)
            total += 1

        return acc1, acc3, acc5, total

    def reset_acc(self):
        self.total = 0
        self.acc1 = 0
        self.acc3 = 0
        self.acc5 = 0

    def get_acc1(self):
        return self.acc1 / self.total

    def get_acc3(self):
        return self.acc3 / self.total / 3

    def get_acc5(self):
        return self.acc5 / self.total / 5

    def get_total(self):
        return self.total



def get_accuracy(logits, labels):
    scores, indices = torch.topk(logits.detach().cpu(), k=10)

    acc1, acc3, acc5, total = 0, 0, 0, 0
    for index, label in enumerate(labels.cpu().numpy()):
        label = set(np.nonzero(label)[0])

        labels = indices[index, :5].numpy()

        acc1 += len(set([labels[0]]) & label)
        acc3 += len(set(labels[:3]) & label)
        acc5 += len(set(labels[:5]) & label)
        total += 1

    return acc1, acc3, acc5, total


def read_configuration(config_file):
    yaml_loader = yaml.FullLoader
    yaml_loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
                 [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\\.(?:inf|Inf|INF)
                |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open(config_file, 'r') as f:
        config_dict = yaml.load(f.read(), Loader=yaml_loader)

    return config_dict


def save(dataset, epoch_idx, mark, plm, tokenizer, *other_models):
    save_dir = f"./checkpoint/{dataset}/{epoch_idx}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_to_save = plm.module if hasattr(plm, 'module') else plm

    model_to_save.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    for model in other_models:
        torch.save(model.module.state_dict() if hasattr(model, 'module') else model, os.path.join(save_dir, "student_model_file.bin"))

    with open(os.path.join(save_dir, "mark.txt"), "w", encoding="UTF-8") as mark_file:
        mark_file.write(mark)

    print(f"save model to path {save_dir} success")
