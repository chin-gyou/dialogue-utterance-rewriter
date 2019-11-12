# dialogue-utterance-rewriter
## dialogue-utterance-rewriter-corpus

Dataset for ACL 2019 paper "[Improving Multi-turn Dialogue Modelling with Utterance ReWriter
](https://arxiv.org/abs/1906.07004)"

After another two months of human labeling, we release a much more better quality dataset(only positive samples) than the original one we used in our paper for better research.

### Description

The positive dataset, 20000 dialogs. Each line in corpus.txt consists of four utterances of dialog (two context utterances, current utterance), and the rewritten uterance. Each line is `tab-delimited` (one tab) with the following format:

```bash
<A: context_1>\t<B: context_2>\t<A: current>\t<A: A: rewritten current>
```

## LSTM-based Model
### About the code
This code is based on the [Pointer-Generator](https://github.com/abisee/pointer-generator) code. 

**Requirements**

To run the souce codes, some external packages are required

* python 2.7
* Tensorflow  1.4

vocab file:
```bash
<word>\t<count>
```
### Run training and Run (concurrent) eval
You may want to run a concurrent evaluation job, that runs your model on the validation set and logs the loss. To do this, run:
To train your model, run:

```
sh train.sh
sh val.sh
```
### Run beam search decoding
To run beam search decoding, first set restore_best_model=1 to restore the best model.

```
sh train.sh
sh test.sh
```
**Why can't you release the Transformer model?** Due to the company legal policy reasons, we cannot realease the Transformer code which has been used in online environment. However, feel free to email us to discuss training and model details. 

### Citation

```
@article{su2019improving,
  title={Improving Multi-turn Dialogue Modelling with Utterance ReWriter},
  author={Su, Hui and Shen, Xiaoyu and Zhang, Rongzhi and Sun, Fei and Hu, Pengwei and Niu, Cheng and Zhou, Jie},
  journal={ACL},
  year={2019}
}
```


