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
