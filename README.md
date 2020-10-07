# *Imagefy*

## *What is It?*
---
- A AI Driven Module that will help you choose the best pictures

## *How Does It do that?*
---
- Imagefy Uses KMeans Clustering & InceptionResNetV2 CNN in order to Group  Images by Similarity and then choose the best image out of every group

## *Recomended setup*
---
- *Make Sure you have the appropriate Drivers for Tensorflow*
- *List of drivers & installtion instructions can be found [here](https://www.tensorflow.org/install/gpu)* 

```
1. git clone https://github.com/yuval7451/Imagefy.git
2. cd Imagefy
3. virtualenv imagefyenv
4. imagefyenv\Scripts\activate
5. pip install -r requirments.txt
```

## Usage & Help
---
#### Imagefy has serval Modules & Loaders
1. Help
```
python3 imagefy\main.py --help
```

2. Generall Usage:
```
imagefy\main.py -d "YOU\DATA\DIR" -o "YOUR\OUTPUT\DIR" -s IMAGE_SIZE -l DATA_LOADER WRAPER_NAME <WRAPER PARAMS>
```
3. MiniBatchKmeans & TensorLoader - (aka tf.data.Dataset DataLoader)
```
imagefy\main.py  -d "YOU\DATA\DIR" -o "YOUR\OUTPUT\DIR" --tensorboard -l tensor_loader mini_kmeans -e <EPOCH> -b <BATCH SIZE> -c <NUM CLUSUSTERS>
```

4. MiniBatchKmeans & DataLoader - (aka My MultiThreader implementation of a DataLoader)
```
imagefy\main.py  -d "YOU\DATA\DIR" -o "YOUR\OUTPUT\DIR" --tensorboard -l data_loader mini_kmeans -e <EPOCH> -b <BATCH SIZE> -c <NUM CLUSUSTERS>
```

5. Kmeans & DataLoader - (aka My MultiThreader implementation of a DataLoader)
```
imagefy\main.py  -d "YOU\DATA\DIR" -o "YOUR\OUTPUT\DIR" --tensorboard -l data_loader kmeans --start <Min number of Clusters> --end <Max number of clisters> -i <Number of training iteration>
```