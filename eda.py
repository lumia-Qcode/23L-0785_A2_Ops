import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

df = pd.read_csv("reports/audit.csv")
ok = df[~df.corrupt]

fig, axs = plt.subplots(2, 6, figsize=(18, 6))
for r, (cls, g) in enumerate(ok.groupby("class")):
    for c, p in enumerate(g.sample(6, random_state=1).path):
        axs[r, c].imshow(Image.open(p))
        axs[r, c].axis("off")
        axs[r, c].set_title(cls if c == 0 else "")
plt.savefig("reports/samples.png", dpi=120, bbox_inches="tight")

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axs, ["width", "height", "size_kb"]):
    for cls, g in ok.groupby("class"):
        ax.hist(g[col], bins=40, alpha=0.6, label=cls)
    ax.set_title(col)
    ax.legend()
plt.savefig("reports/hists.png", dpi=120, bbox_inches="tight")