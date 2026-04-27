

# MED-HIDE

Medical Image Data Hiding using Deep Learning and Adversarial Robustness

---

## 📦 Requirements

```bash
pip install torch torchvision numpy matplotlib pillow tqdm reedsolo
```

---

## 📁 Dataset

Download dataset from:
[https://www.kaggle.com/datasets/anavisrivastava/brain-tumor-mri-ct-medical-images-pneumonia-xray](https://www.kaggle.com/datasets/anavisrivastava/brain-tumor-mri-ct-medical-images-pneumonia-xray)

After downloading and extracting, update the path in `Config`:

```python
DATA_PATHS = [
    "path/to/brain-tumor-mri-ct-medical-images-pneumonia-xray"
]
```

---

## ▶️ Run Code

```bash
python main.py
```

---

## 💾 Output

After training, weights will be saved as:

```
stego_enc_gan.pth
stego_dec_gan.pth
```

---

## ▶️ Using Saved Weights

```python
encoder.load_state_dict(torch.load("stego_enc_gan.pth"))
decoder.load_state_dict(torch.load("stego_dec_gan.pth"))
```

---

## ⚠️ Notes

* Uses GPU if available
* Default image size: 256×256
* Modify parameters inside `Config` if needed

