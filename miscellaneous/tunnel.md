# SSH Port Forwarding aka SSH tunnel

SSH Tunnel from `localhost` to `HOST`

from `localhost`:

```bash
ssh -L 9999:localhost:9999 HOST
```

## How to open jupyter notebook on a remote host

go to `HOST` and open jupyter notebook

```bash
jupyter notebook --port=9999 --no-browser
```

To access the notebook, open a browser on `localhost` and copy and paste the URL


# SSH tunnel via multiple hops

SSH Tunnel from localhost to `HOST1` and from `HOST1` to `HOST2`

from `localhost`:

```bash
ssh -L 9999:localhost:9999 HOST1 ssh -L 9999:localhost:1234 -N HOST2
```

go to `HOST2` and open jupyter notebook

```bash
jupyter notebook --port=9999 --no-browser
```

# Useful Tips

to kill process running on a port
```bash
lsof -ti:9999 | xargs kill -9
```
