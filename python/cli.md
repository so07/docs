# Python Command Line Interface

## ```argparse``` simple example

```python
import argparse

parser = argparse.ArgumentParser(
    prog="prog", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
)

parser.add_argument(
    "--version",
    action="version",
    version="0.0",
    help="version",
)

parser.add_argument(
    "-v", "--verbose", action="count", default=0, help="increase verbosity"
)

parser.add_argument("-a", "--argument", dest="argument", help="argument help")

args = parser.parse_args()
```
