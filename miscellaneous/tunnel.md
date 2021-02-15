# SSH tunnel via multiple hops

SSH Tunnel from localhost to HOST1 and from HOST1 to HOST2

from ```localhost```

```bash
ssh -L 9999:localhost:9999 HOST1 ssh -L 9999:localhost:1234 -N HOST2
```

go to ``HOST2``` and open jupyter notebook

```bash
jupyter notebook --port=9999 --no-browser
```

