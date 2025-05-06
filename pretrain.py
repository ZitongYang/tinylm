import os
from datasets import load_dataset, Dataset, concatenate_datasets
from pprint import pprint
import pdb
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def streaming():
    """
    Pretraining dataset is huge, e.g. 15T tokens.
    There is no way to load it into any memory or event disk.
    Need some form of streaming processing.
    """
    ds = load_dataset("mlfoundations/dclm-baseline-1.0-parquet",
                      split="train",
                      trust_remote_code=True,
                      streaming=True)
    for example in ds:
        pprint(example)
        pdb.set_trace()

def _contains_statistics(example: dict) -> bool:
    return "statistics" in example["text"]

def parallelism():
    """
    Consider a the task of searching for documents contain "statistics"
    This is naturally parallelizable, as with many preprocessing steps
    """
    ds = load_dataset("mlfoundations/dclm-baseline-1.0-parquet",
                      split="train",
                      trust_remote_code=True,
                      streaming=True)
    ds = ds.take(1000)
    ds = Dataset.from_list(list(ds))
    print(f"Current length: {len(ds)}")
    pdb.set_trace()
    ds = ds.filter(_contains_statistics,
                   num_proc=os.cpu_count())
    print(f"After filtering: {len(ds)}")
    for example in ds:
        pdb.set_trace()
        pprint(example)

def _process_shard(shard_index: int, total_shards: int):
    ds = load_dataset("mlfoundations/dclm-baseline-1.0-parquet",
                      split="train",
                      trust_remote_code=True,
                      streaming=True)
    ds = ds.shard(num_shards=total_shards, index=shard_index)
    ds = ds.take(500)
    ds = ds.filter(_contains_statistics)
    ds = Dataset.from_list(list(ds))
    return ds

def combined(approach: str):
    """
    Streaming dataset is naturally non-parallelizable
    But should we integrate both world?

    # Sharding
    ['today...', 'hello...', 'world...', 'statistics...', ...]
    --------shard 1--------   ------------shard 2-------------
    1.  Split the dataset into N shards
    2.  Process each shard with a different process
    3.  Merge the shards at the end
    """
    if approach == "sharding":
        map_process_shard = partial(_process_shard, total_shards=2)
        import pdb; pdb.set_trace()
        with ProcessPoolExecutor(max_workers=2) as executor:
            processed = executor.map(map_process_shard,
                                     [0, 1])
            processed = list(processed)
        ds = concatenate_datasets(processed)
        for example in ds:
            pprint(example)
            pdb.set_trace()
    elif approach == "interleave":
        note = (
        """
        ['today...', 'hello...', 'world...', 'statistics...', ...]
        ---ps1---  ----ps2----    ---ps1--   ------ps2-----

        More robust, industrial-strength way
        - autoscaling
        - error handling
        - "exists one successful processing" programming model
        Softwares:
        - MapReduce (Jeff Dean, apache_beam)
        - Spark (Berkeley, apache_spark)
        """)
        print(note)
    else:
        raise ValueError(f"Invalid approach: {approach}")

if __name__ == "__main__":
    streaming()
    # parallelism()
    # combined(approach="sharding")
    # combined(approach="interleave")