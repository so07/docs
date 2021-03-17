# `subprocess`

execute command and print stdout

```python
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
