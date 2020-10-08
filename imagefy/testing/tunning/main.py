
import time
# import logging; logging.getLogger(__name__).setLevel(logging.INFO)
from tabulate import tabulate
from imagefy.utils.common import BASE_PATH_DEST, BATCH_SIZE_DEST, DIR_DEST, EPOCHS_DEST, NUM_CLUSTERS_DEST, SIZE_DEST, TENSORBOARD_DEST, TOP_DEST, VERBOSE_DEST
from imagefy.suits.intergration_suit import IntergrationSuit
from pprint import pformat

def tune():
    image_sizes = [64, 124, 224, 521]
    num_epochs = [1,2]
    batch_sizes = [20, 40, 80]
    num_clusters = [4, 6, 8]
    tensorboard = True
    base_path = 'D:\\Imagefy\\results'
    dir_path = 'D:\\Imagefy\\dataset\\small\\*'
    top = 3
    verbose = False
    params_list = {
        "total_time": [],
        "score" : [],
        "model_name": [],
        DIR_DEST: [],
        SIZE_DEST: [],
        EPOCHS_DEST: [],
        BATCH_SIZE_DEST: [],
        NUM_CLUSTERS_DEST: [],
    }

    for num_epoch in num_epochs:
        for batch_size in batch_sizes:
            for num_cluster in num_clusters: 
                for image_size in image_sizes:
                    params = {
                        DIR_DEST: dir_path,
                        SIZE_DEST: image_size,
                        TENSORBOARD_DEST: tensorboard,
                        BASE_PATH_DEST: base_path,
                        EPOCHS_DEST: num_epoch,
                        BATCH_SIZE_DEST: batch_size,
                        NUM_CLUSTERS_DEST: num_cluster,
                        TOP_DEST: top,
                        VERBOSE_DEST: verbose,
                    }
                    print(f"Params are {pformat(params)}")
                    start = time.time()
                    suit = IntergrationSuit(**params)
                    suit.run()
                    total_time  = time.time() - start
                    model_name = suit.model_name
                    score = suit.kmeans.score
                    params_list[SIZE_DEST].append(image_size)
                    params_list[DIR_DEST].append(dir_path)
                    params_list[EPOCHS_DEST].append(num_epoch)
                    params_list[BATCH_SIZE_DEST].append(batch_size)
                    params_list[NUM_CLUSTERS_DEST].append(num_cluster)
                    params_list["total_time"].append(total_time)
                    params_list["model_name"].append(model_name)
                    params_list["score"].append(score)
                    print(tabulate(params_list, tablefmt='pretty', headers=params_list.keys()))

    return params_list
                    
                    


if __name__ == '__main__':
    start = time.time()
    params_list = tune()
    end = time.time() - start
    with open("HPTunning.txt", "w") as fd:
        fd.write(tabulate(params_list, tablefmt='pretty', headers=params_list.keys()))
    print(f"It Took {end} seconds to complete")