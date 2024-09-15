This dataset provides masked sentences and multi-token phrases that were masked-out of these sentences.
 
### Experiment_1 ###
We offer 3 datasets: a training/dev data extracted from the Wikipedia, and 2 additional test datasets extracted from PDTB-3 and DiscoGeM 1.0. 
For these datasets, the columns provided for each datapoint are as follows:

Training/dev dataset:
text- the original sentence
span- the multi-token conenctive which is masked out
span_lower- the lowercase version of span
range- the range in the text string which will be masked out (this is important because span might appear more than once in text)
freq- the corpus frequency of span_lower
masked_text- the masked version of text, span is replaced with [MASK]

Test datasets:
corpus- PDTB-3/DiscoGeM
datasource- from which file in the corpus
genre- the genre that the text comes from.
text- the original discourse (Arg1 and Arg2)
connective- the implicit conenctive which is masked out
range- the range in the text string which will be masked out (this is important because span might appear more than once in text)
sense- the gold label
preposed_phrase- the preposed constituent in the original text
masked_text- the masked version of text, connective is replaced with [MASK]