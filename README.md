# Learning to Learn how to Learn: Self-Adaptive Visual Navigation using Meta-Learning

By Mitchell Wortsman, Kiana Ehsani, Mohammad Rastegari, Ali Farhadi and Roozbeh Mottaghi (Oral Presentation at CVPR 2019).


[CVPR 2019 Paper ](https://arxiv.org/abs/1812.00971) | [Video](https://www.youtube.com/watch?v=-Ba6ZRMcxEE&feature=youtu.be) | [BibTex](#citing)

Intuition            |  Examples
:-------------------------:|:-------------------------:
![](figs/abstract_figure.jpg)  |  ![](figs/qualitative.jpg)

There is a lot to learn about a task by actually attempting it! Learning is continuous, i.e. we learn as we perform.
Traditional navigation approaches freeze the model during inference (top row in the intuition figure above). 
In  this  paper,  we  propose a self-addaptive agent for visual navigation that learns via self-supervised
interaction with the environment (bottom row in the intuition figure above).


## Citing

If you find this project useful in your research, please consider citing:

```
@InProceedings{Wortsman_2019_CVPR,
  author={Mitchell Wortsman and Kiana Ehsani and Mohammad Rastegari and Ali Farhadi and Roozbeh Mottaghi},
  title={Learning to Learn How to Learn: Self-Adaptive Visual Navigation Using Meta-Learning},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019}
}
```

## Results


| Model  | SPL  &geq; 1 | Success  &geq; 1 | SPL   &geq; 5 | Success  &geq; 5 |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| [SAVN](#SAVN)  |  16.15  &pm; 0.5 | 40.86  &pm; 1.2 | 13.91  &pm; 0.5 | 28.70  &pm; 1.5 |
| [Scene Priors](https://arxiv.org/abs/1810.06543)  | 15.47  &pm; 1.1 | 35.13  &pm; 1.3 | 11.37  &pm; 1.6 | 22.25  &pm; 2.7 |
| [Non-Adaptive A3C](#Non-Adaptvie-A3C)  | 14.68  &pm; 1.8 | 33.04  &pm; 3.5 | 11.69  &pm; 1.9 | 21.44  &pm; 3.0 |


## Setup

- Clone the repository with `git clone https://github.com/allenai/savn.git && cd savn`.

- Install the necessary packages. If you are using pip then simply run `pip install -r requirements.txt`.

- Download the [pretrained models](https://prior-datasets.s3.us-east-2.amazonaws.com/savn/pretrained_models.tar.gz) and
[data](https://prior-datasets.s3.us-east-2.amazonaws.com/savn/data.tar.gz) to the `savn` directory. Untar with
```bash
tar -xzf pretrained_models.tar.gz
tar -xzf data.tar.gz
```

The `data` folder contains:

- `thor_offline_data` which is organized into sub-folders, each of which corresponds to a scene in [AI2-THOR](https://ai2thor.allenai.org/). For each room we have scraped the [ResNet](https://arxiv.org/abs/1512.03385) features of all possible locations in addition to a metadata and [NetworkX](https://networkx.github.io/) graph of possible navigations in the scene.
- `thor_glove` which contains the [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings for the navigation targets.
- `gcn` which contains the necessary data for the [Graph Convolutional Network (GCN)](https://arxiv.org/abs/1609.02907) in [Scene Priors](https://arxiv.org/abs/1810.06543), including the adjacency matrix.

Note that the starting positions and scenes for the test and validation set may be found in `test_val_split`.

If you wish to access the RGB images in addition to the ResNet features, replace `thor_offline_data` with [thor_offlline_data_with_images](https://prior-datasets.s3.us-east-2.amazonaws.com/savn/offline_data_with_images.tar.gz). If you wish to run your model on the image files,
add the command line argument `--images_file_name images.hdf5`. 

## Evaluation using Pretrained Models

Use the following code to run the pretrained models on the test set. Add the argument `--gpu-ids 0 1` to speed up the evaluation by using GPUs.

#### SAVN
```bash
python main.py --eval \
    --test_or_val test \
    --episode_type TestValEpisode \
    --load_model pretrained_models/savn_pretrained.dat \
    --model SAVN \
    --results_json savn_test.json 

cat savn_test.json 
```

#### Scene Priors
```bash
python main.py --eval \
    --test_or_val test \
    --episode_type TestValEpisode \
    --load_model pretrained_models/gcn_pretrained.dat \
    --model GCN \
    --glove_dir ./data/gcn \
    --results_json scene_priors_test.json

cat scene_priors_test.json 
```


#### Non-Adaptvie-A3C
```bash
python main.py --eval \
    --test_or_val test \
    --episode_type TestValEpisode \
    --load_model pretrained_models/nonadaptivea3c_pretrained.dat \
    --results_json nonadaptivea3c_test.json

cat nonadaptivea3c_test.json
```

The result may vary depending on system and set-up though we obtain:

| Model  | SPL  &geq; 1 | Success  &geq; 1 | SPL   &geq; 5 | Success  &geq; 5 |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| [SAVN](#SAVN)  |  16.13 | 42.20 | 14.30 | 30.09 |
| [Scene Priors](https://arxiv.org/abs/1810.06543)  |  14.86 | 36.90 | 11.49 | 24.70 |
| [Non-Adaptive A3C](#Non-Adaptvie-A3C)  | 14.10 | 32.40 | 10.73 | 19.16 |

The results in the [initial submission](https://arxiv.org/abs/1812.00971v1) (shown below) were the best (in terms of success on the validation set). After the initial submission, we trained the model 5 times from scratch to obtain error bars, which you may find in [results](#results).

| Model  | SPL  &geq; 1 | Success  &geq; 1 | SPL   &geq; 5 | Success  &geq; 5 |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| [SAVN](#SAVN)  |  16.13 | 42.10 | 13.19 | 30.54 |
| [Non-Adaptive A3C](#Non-Adaptvie-A3C)  | 13.73 | 32.90 | 10.88 | 20.66 |

## How to Train your SAVN

You may train your own models by using the commands below.

#### Training SAVN
```bash
python main.py \
    --title savn_train \
    --model SAVN \
    --gpu-ids 0 1 \
    --workers 12
```


#### Training Non-Adaptvie A3C
```bash
python main.py \
    --title nonadaptivea3c_train \
    --gpu-ids 0 1 \
    --workers 12
```


## How to Evaluate your Trained Model

You may use the following commands for evaluating models you have trained.

#### SAVN
```bash
python full_eval.py \
    --title savn \
    --model SAVN \
    --results_json savn_results.json \
    --gpu-ids 0 1
    
cat savn_results.json
```

#### Non-Adaptive A3C
```bash
python full_eval.py \
    --title nonadaptivea3c \
    --results_json nonadaptivea3c_results.json \
    --gpu-ids 0 1
    
cat nonadaptivea3c_results.json
```

####  Random Agent
```bash
python main.py \
    --eval \
    --test_or_val test \
    --episode_type TestValEpisode \
    --title random_test \
    --agent_type RandomNavigationAgent \
    --results_json random_results.json
    
cat random_results.json
```