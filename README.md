# tinylm
> Large-scale maximum likelihood estimation with non-parametric models

## Prelude: few-shot learning
> Reservoir of machine learnable knowledge: 95% unsupervised learning and 5% supervision.
#### An intriguing phenomenon around 2017
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
- Representation learning, searching for weaker forms of supervision (e.g., Hinton's SimCLR)
- Claim: the GPT-3 revolution is foreshadowed by this phenomenon

## Outline
- ~~Prelude: few-shot learning~~
- Pretraining (training on non-cat/dog images)
    - Data
        - Crawler
        - System design concepts
    - Architecture comments
- Posttraining (training on cat/dog images)
    - Instruction tuning data
        - Human annotation
    - Trainer
- Evaluation (can model distinguish cat vs. dog?)
    - Human evaluation
- Final remarks

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
- Streaming
- Parallelism
- Combined

#### Architecture comments
Variation of transformer architecture exists. Notably,
- Decoder-only
- Pre-norm/Post-norm. Batch/Layer/RMS-norm
- Position embedding
- MHA, GQA, MLA, ...

System consideration:
- Kernel optimization
- Mixed precision
- Falut tolerance
- Checkpointing
- Gradient accumulation
- ...

As a product of pretraining, we get a text completion model
```
LM("The captial of France is") -> "Paris"
```

## Posttraining
> Ultimately, we want model to answer to user instructions. For example LM("What is the capital of France?") -> "Paris".

