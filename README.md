# tinylm
> Large-scale maximum likelihood estimation with non-parametric models

## Prelude: few-shot learning
> Reservoir of machine learnable knowledge: 95% unsupervised learning and 5% supervision.

#### An intriguing phenomenon around 2016+
Problem setup:
- Image classification task with logistic loss
- Goal: train neural network to classify `cat` vs. `dog`

Approach I:
- Collect all `(image, label)` pair whose label is `cat` or `dog`
- Performn standard training pipeline
- Needs 5,000 samples to get good accuracy on `cat` vs. `dog`

Approach II:
- Collect all `(image, label)` pair whose label is *not* `cat` or `dog`
- Train the network on this dataset.
- Test the network on `cat` vs. `dog`. Attain almost 0% accuracy.
- Train the network on a *few* (less than 10) `(image, label)` pair with `cat` vs. `dog`
- Test the network on `cat` vs. `dog` recover high accuracy.

#### Interpretation
- Training on non-cat/dog images is a form of unsupervised learning that are not directly useful for classifying cat vs. dog
- They prepare the weights of the model in an *activated* state for new tasks
- In this *activated* state, the model is able to learn the task with a few samples.

#### How is this picked up by the community?
- Representation learning, searching for weaker forms of supervision (e.g., SimCLR)
- Claim: the GPT-3 revolution is foreshadowed by this phenomenon

## Outline
- ~~Prelude: few-shot learning~~
- Pretraining (training on non-cat/dog images)
    - Data
        - Crawler
        - System design concepts
    - Architecture comments
- Posttraining (training on cat/dog images)
    - Instruction following dataset
    - Supervised finetuning
- Final remarks
    - Evaluation (can model distinguish cat vs. dog?)
    - Take-home message

## Pretraining
> Prepares the model in an *activated* state that can be efficiently adapted to new tasks.

### Data
Next token prediction on all internet text

#### Crawler
How should I visit all the webpages?
- search a lot of queries on Google?
- try all http requests and see what's out there?
- focus on specific subsets, e.g., wikipedia?

GPT-2: a revolution
- Iterate through all reddit posts. Create the dictionary of `{webpage_link: sum_of_reddit_likes_on_posts_that_refer_to_this_webpage}`.
- Higher score means higher quality webpages
- Train the model on these high-quality webpages

Idea: internet is a directed graph. Crawlers try to traverse the graph as much as possible.
- Underlooked: accademia just takes whatever on huggingface
- A lot of opportunities for algorithmic and statistical advancement: joint search and ranking.

#### System design concepts
- Streaming[[code]](pretrain.py#L8)
- Parallelism[[code]](pretrain.py#L25)
- Combined[[code]](pretrain.py#L56)

#### Architecture comments
Variation of transformer architecture exists. Notably,
- Decoder-only
- Pre-norm/Post-norm. Batch/Layer/RMS-norm
- Position embedding
- MHA, GQA, MLA, ...

System consideration:
- Kernel optimization
- Mixed precision
- Fault tolerance
- Checkpointing
- Gradient accumulation
- ...

As a product of pretraining, we get a text completion model[[code]](decoding.py#L10)
```
LM("The captial of France is") -> "Paris"
```

## Posttraining
> Ultimately, we want model to answer to user instructions. For example LM("What is the capital of France?") -> "Paris".

### Instruction following dataset
Two stage process
- Query curation `x`: how to generate truly diverse instructions?
    - Iterate through the pretraining data
    - Ask human annotator to design querys that would have the given pretraining document as output
- Human annotation `y`: how to teach model the desired behavior?
    - Ask human annotators to respond to the query

Finally, prepare the dataset in a chat template[[code]](dataloader.py#L30).

### Supervised finetuning
Perform next-token prediction on the instruction following dataset [[code]](train.py#L26).

## Final remarks

### Evaluation
- Early days (Breiman's second culture), held-out test set. Train on $L_n(\theta)$ and evaluate on $\mathbb{E}[L_1(\theta)]$
- Current practice, held-out "task". Pretrain on internet text, finetune on instruction following dataset and test on SAT, ACT, AIME, IMO, etc.
- Future, language model are fundementally products (e.g, cars, phones, etc). They should be evaluated based on user experience in the same way any other commodities are evaluated (e.g., mpg, engine size, etc.).

### Take-home message
> Pretraining is regularization

Linear regression:

$\min_{\theta} \|X\theta-y\|^2+ \lambda \|\theta\|^2$

Language model:

$\min_{\theta} L_{\text{instruction-following}}(\theta) + \lambda L_{\text{internet-texts}}(\theta)$

In words,
- Directly performing instruction following (which is ultimately what we want), the LM overfits to the instruction set.
- Pretraining regularizes the space of transformer weights to be around natural language, making it easier to generalize to new instructions.
