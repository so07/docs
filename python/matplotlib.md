# Matplotlib

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot(x, y, lw=1, ls="--", c="black", alpha=0.5)
ax.plot(x, y, lw=2, c="#1f77b4", marker="o", markersize="5")

ax.set_xlabel("x")
ax.set_ylabel("y")

ax.grid(b=True, which='major', axis='y', color='gray', alpha=0.25, linestyle='-')

ax.legend()

plt.savefig("fig.png", transparent=False)
```
