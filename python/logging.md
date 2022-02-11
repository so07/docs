# `logging` module

```python
logging.basicConfig(
    level=logging.WARNING,
    format= '[%(asctime)s] [%(pathname)s:%(lineno)d] %(levelname)s - %(message)s',
    #format="#[%(asctime)s] >>> %(message)s"
    datefmt='%H:%M:%S'
)
```

```python
import logging

logger = logging.getLogger()

def set_logging_verbosity(verbose):
    level = logging.INFO
    if verbose:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format= '[%(asctime)s] [%(pathname)s:%(lineno)d] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )    
```

```python
import logging

def logger(name, verbose=False):
    logging.basicConfig(
        level=logging.INFO,
        format= '[%(asctime)s] [%(pathname)s:%(lineno)d] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger(name)
    if verbose:
        logger.setLevel(logging.DEBUG)
    return logger

logger = logger(__name__, True)
```

