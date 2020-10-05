# *Imagefy*

## *What is It?*
---
- A AI Driven Module that will help you choose the best pictures

## *How Does It do that?*
---
- Imagefy Uses KMeans Clustering & InceptionResNetV2 CNN in order to Group  Images by Similarity and then choose the best image out of every group

## *How Can i Use it to?*
---
### Imagefy has serval Modules & Loaders
1. Generall Usage:
```
integration\main.py -d <DATA DIR> -o <OUTPUT DIR> -s <IMAGE SIZE> -l <DATA LOADER> <!WRAPER> <WRAPER PARAMS>
```
2. MiniBatchKmeans & TensorLoader - (aka tf.data.Dataset DataLoader)
```
integration\main.py -d <DATA DIR> -v <!VERBOSE> -o <OUTPUT DIR> -s <IMAGE SIZE> --tensorboard <!LOG TENSOORBOARD> -l tensor_loader <!DATA LOADER> mini_kmeans <!WRAPER> -e <EPOCH> -b <BATCH SIZE> -c <NUM CLUSUSTERS>
```

---
# Packegs & Versions
- Python ~3.7
- Cuda ToolKit==10.0
- CuDNN==7.6.5
- tensorflow-gpu==1.15
- keras==2.2.5?
- opencv-python
- scikit-learn
- matplotlib