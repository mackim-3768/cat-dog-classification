# cat-dog-classification
This Project contains a complete pipeline to train a simple Cats vs Dogs classifier, export it to ExecuTorch targeting Samsung ENN backend (E9955), run it on an Android device via ADB, and visualize results as an overlay on the input image.

## 1) Train and Export (.pte)
```bash
python catsdogs_train_export.py \
  --data-root ./cats_and_dogs_filtered \
  --artifact ./artifacts_catsdogs \
  --epochs 3 --batch-size 32 --chipset E9955
```

## 2) Run on Device and Create Overlay
```bash
python catsdogs_infer_overlay.py \
  --image ./cats_and_dogs_filtered/validation/dogs/dog.2000.jpg \
  --model ./artifacts_catsdogs/catsdogs_mobilenetv2.pte \
  --labels ./artifacts_catsdogs/labels.txt \
  --artifact ./artifacts_catsdogs_infer \
  --runner ../../executorch/build_samsung_android/backends/samsung/enn_executor_runner \
  --serial <device_serial> \
  --topk 2
```
