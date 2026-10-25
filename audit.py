import os, hashlib
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = "data/raw"         
AI_FOLDER = "AiArtData"  
os.makedirs("reports", exist_ok=True)
os.makedirs("splits", exist_ok=True)

rows = []
for cls in sorted(os.listdir(ROOT)):
    for dp, _, files in os.walk(os.path.join(ROOT, cls)):
        for f in files:
            p = os.path.join(dp, f)
            with open(p, "rb") as fh:
                sha = hashlib.sha256(fh.read()).hexdigest()
            r = {"path": p, "class": cls, "size_kb": os.path.getsize(p) / 1024, "sha256": sha}
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    r.update(width=im.width, height=im.height, mode=im.mode, format=im.format, corrupt=False)
            except Exception:
                r.update(width=None, height=None, mode=None, format=None, corrupt=True)
            rows.append(r)

df = pd.DataFrame(rows)
df["duplicate"] = df.duplicated("sha256", keep="first")
df.to_csv("reports/audit.csv", index=False)
print(df.groupby("class").size())
print(df.groupby("class")[["width", "height", "size_kb"]].agg(["min", "median", "max"]))
print("corrupt:", int(df.corrupt.sum()), "duplicates:", int(df.duplicate.sum()))

clean = df[(~df.corrupt) & (~df.duplicate)].copy()
clean["label"] = (clean["class"] == AI_FOLDER).astype(int)   # 1 = AI-generated
train, tmp = train_test_split(clean, test_size=0.3, stratify=clean.label, random_state=42)
val, test = train_test_split(tmp, test_size=0.5, stratify=tmp.label, random_state=42)
for name, d in [("train", train), ("val", val), ("test", test)]:
    d[["path", "label"]].to_csv(f"splits/${name}.csv", index=False)
print(len(train), len(val), len(test))