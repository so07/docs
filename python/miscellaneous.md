# `subprocess`

execute command and print stdout

```python
import subprocess

def run(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    stdout = ""
    while True:
        line = p.stdout.readline()
        if line == b'' and p.poll() is not None:
            break
        if line:
            line_str = line.decode('utf-8')
            stdout += line_str
            print(line_str.strip())
    rc = p.poll()
    return rc, stdout.strip()
```

# `logging`

```python
import logging

logger = logging.getLogger()

def set_logging_verbosity(verbose):
    level = logging.INFO
    if verbose:
        level = logging.DEBUG
    logging.basicConfig(level=level)
    #logging.basicConfig(format="#[%(asctime)s] >>> %(message)s", level=level)
```

```
logging.basicConfig(
    level=logging.WARNING,
    format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
```
