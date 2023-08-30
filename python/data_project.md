# Example of data project

import all modules

```python
import os
import re
import glob
import logging

import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple
```

## Logger

```python
verbose = False

logging.basicConfig(
    level=logging.INFO,
    format= '# [%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger()

if verbose:
    logger.setLevel(logging.DEBUG)
```


## Data structure

```python
Data = namedtuple("Data", "label x y z") # MODIFY DATA

logger.debug(f"data fields: {Data._fields}")
```


## Parser

```python
def parser(path, label="", suffix="log", sort_by="x"): # MODIFY SORT_BY OPTION

    # list files
    ls = glob.glob(os.path.join(path, f"*.{suffix}"))

    logger.info(f"found {len(ls)} files in {path}")
    logger.debug(f"files: {ls}")

    # read data
    ld = []
    for f in ls:
        # read log
        with open(f) as fp:
            log = fp.read()
        # parse log
        d = re.search("(?:TOKEN )(.+)(?: x )(.+)(?: y )(.+)(?: z)", log) # MODIFY PARSER
        logger.debug(f"match: {d}")
        if d:
            d = d.groups()
            logger.debug(f"groups: {d}")
            d = Data(label, int(d[0]), float(d[1]), float(d[0])) # MODIFY DATA TYPE
            logger.debug(f"data: {d}")
            ld.append(d)

    logger.debug(f"found data: {ld}")

    # sort by a key
    ld = sorted(ld, key=lambda k: getattr(k, sort_by))
    logger.debug(f"sorted data: {ld}")

    # from a list of data to dict of list
    d = {k: np.array([getattr(i, k) for i in ld]) for k in Data._fields}
    logger.debug(f"data dict: {d}")

    # from dict to namedtuple
    d = Data(label, *[d[k] for k in Data._fields[1:]])
    logger.debug(f"return data: {d}")

    return d
```

## Read data

```python
l = [ # MODIFY PATH LIST
    ("path1", "key1"),
    ("path2", "key2"),
    ("path3", "key3"),
]

data = [parser(*i) for i in l]

logger.info(f"found {len(data)} data")
logger.debug(f"all data: {data}")
```


## Plot

```python
fig, ax = plt.subplots()

for d in data:

    ax.plot(d.x, d.y, lw=1.25, label=d.name) # MODIFY PLOT
    ax.legend()

    ax.set_yscale('log')

plt.savefig("fig.png", transparent=False)
```

## Convert to DataFrame

```python
import pandas as pd

def create_dataframe(data):
    df = pd.DataFrame()
    for d in data:
        _df = pd.DataFrame.from_dict({k: getattr(d, k) for k in Data._fields})
        df = pd.concat([df, _df], ignore_index=True)
    return df
```

```python
from tabulate import tabulate

def print_df(df, title=None):
    """print dataframe in fancy mode"""
    if title is not None:
        print(title)
    print(tabulate(df, headers='keys', tablefmt='psql', numalign="right", floatfmt=".2f", showindex=False))
```
