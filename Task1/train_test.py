from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings, BertEmbeddings, ELMoEmbeddings,RoBERTaEmbeddings
from typing import List
from flair.visual.training_curves import Plotter
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import torch
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from pathlib import Path

import logging

def pred2label(pred, X_word_val):
    out_word = []
    out = []
    for pred_i, sent in zip(pred,X_word_val):
        out_i = []
        out_w = []
        s_flag = 0
        for tt, word  in zip(pred_i,sent):                 
            if len(word.text)<3 and not word.text.isalnum():
                if tt == 'B-CHE':
                    s_flag += 1
                continue
               
            if s_flag == 0 :
                out_i.append(tt)
            else:
                out_i.append('B-CHE')
                s_flag = 0
            out_w.append(word.text)
        out.append(out_i)
        out_word.append(out_w)
    return out,out_word

def main():
	# Load Data
    columns = {0: 'text', 1: 'pos', 2: 'ner'}
    data_folder = 'Data'
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file='train.txt',
                                  test_file='test_v2.txt',
                                  dev_file='dev_v2.txt', in_memory=False)

    
    tag_type = 'ner'

    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    # Select word embeddings
    roberta_embedding = RoBERTaEmbeddings()
    #glove_embedding = WordEmbeddings('glove')
    #elmo_embedding = ELMoEmbeddings('pubmed')

    #pos_embedding = WordEmbeddings('pos_v2.gensim', 'pos')
    #bionlp_embedding = WordEmbeddings('bionlp.gensim')
    #biobert_embedding = BertEmbeddings(bert_model_or_path="biobert")
    # init Flair forward and backwards embeddings
    #flair_embedding_forward = FlairEmbeddings('news-forward')
    #flair_embedding_backward = FlairEmbeddings('news-backward')

    embeddings: StackedEmbeddings = StackedEmbeddings([
                                        #bionlp_embedding,
                                        #pos_embedding
                                        roberta_embedding#pubmed_embedding
                                        #biobert_embedding
                                        #elmo_embedding
                                        #glove_embedding,
                                        #flair_embedding_forward,
                                        #flair_embedding_backward,
                                       ])

	# Model for training
    tagger: SequenceTagger = SequenceTagger(hidden_size=256,rnn_layers=2,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)

    

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)
    
    # Start training
    trainer.train('Model/roberta/',
                  learning_rate=0.1,
                  mini_batch_size=32,
                  max_epochs=60,
                  checkpoint=True)
   
    # Start evaluation on test set    
    model = SequenceTagger.load('Model/roberta/best-model.pt')
    gold_all = []
    pred_all = []   
    token_all = []
	
    for sentence in corpus.test:
        tokens  = sentence.tokens
        gold_tags = [token.get_tag('ner').value for token in sentence.tokens]

        tagged_sentence = Sentence()
        tagged_sentence.tokens = tokens

        model.predict(tagged_sentence)

        predicted_tags = [token.get_tag('ner').value for token in tagged_sentence.tokens]

        assert len(tokens) == len(gold_tags)
        assert len(gold_tags) == len(predicted_tags)
                
        gold_all.append(gold_tags)
        pred_all.append(predicted_tags)
        token_all.append(tokens)
              
    pred_labels,cl_word = pred2label(pred_all,token_all)
    test_labels,_ = pred2label(gold_all, token_all)
    
	# Print results
	print('Strict mode results')
    print(classification_report(gold_all, pred_all,digits=3)) 
	print('Relaxed mode results')
    print(classification_report(test_labels, pred_labels, digits=3))

    
if __name__ == '__main__':
    main()

