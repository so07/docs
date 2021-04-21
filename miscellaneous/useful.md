# `screen`

`screen` is a window manager that multiplexes a physical terminal between several processes with terminal emulation.

Start a new session
```bash
screen
```

List of all sessions
```bash
screen -ls
```

Attach to a not detached screen session
```bash
screen -x SESSION_ID
```

# `tmux`

Start a new session
```bash
tmux
```

Create a new named session
```bash
tmux new -s NAME
```
List of all sessions
```bash
tmux ls
```

Attaching to a Tmux Session
```bash
tmux attach-session -t ID
```
```bash
tmux a -t ID
```

Detaching from a Tmux Session
```bash
tmux detach
```

Kill the session
```bash
tmux kill-session –t ID
```


# `/proc`

Every process in linux as a PID (process identifier), an integer number to uniquely identify an active process.\
In `/proc/PID` there are a lot of info about process.

- `/proc/PID/cmdline` \
  command line arguments the process was started with
- `/proc/PID/environ` \
  all process's environment variables
- `/proc/PID/exe` \
  symbolic link to the process executable
- `/proc/PID/status` \
  Info about memory, status, ... on the process
- `/proc/PID/fd` \
  all files opened by process
- `/proc/PID/stack` \
  kernel current stack of process
  
see `man proc` for more info

