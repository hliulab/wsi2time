# Annotation-free pathology images characterize tumor immune microenvironment in breast cancer by weakly supervised contrastive learning  

Firstly, [adversarial contrastive learning](https://arxiv.org/abs/2011.08435) and [Masked Autoencoders](https://arxiv.org/abs/2111.06377) is used to unsupervised extract tile-level features, 
then the attention-pooling is used to aggregate tile-level features into slide-level features, 
and finally it is used in the downstream tumor diagnosis and differential gene expression prediction tasks.


## Segmentation and Patching
You can download your own wsi dataset to the directory slides, 
then run data_processing/create_patches_fp.py to segment and split WSIs,  adjust the parameters according to your needs.  
For example, you can use following command for segment and tile.  
``` shell
python create_patches_fp.py --source ../slides/TCGA-BRCA --data_type tcga_BRCA --save_dir ../tile_results --patch --seg
```  
When you run this command, it will run in default parameter, if you want to run with your parameter, you can modify tcga_lung.csv in directory preset, and add ```--preset ../preset/tcga_BRCA.csv```.
Then the coordinate files will be saved to ```tile_results/patches``` 
and the mask files that show contours of slides will be saved to ```tile_results/masks```.

## Training Contrastive learning model (AdCo) and masked autoencoder (MAE)
Run train/train_adco.py to train contrast learning model on tiles,
you should write Adco/ops/argparser.py to configure the data source
and the save address and ADCO related parameters firstly.
In addition, you need to prepare a CSV file similar to dataset_csv/sample_data.csv,
this file needs to save the name of the WSI file used for training.  
For example, you can use following command for training ADCO model with default parameter.  
``` shell
python train_adco.py --csv_path ../dataset_csv/sample_data.csv --save_path ../MODELS --data_h5_dir ../tile_result --data_slide_dir ../slides/TCGA-LUNG --data_type tcga_lung
```  

Run train/train_mae.py to train contrastive learning model on millions of tiles,
you need to prepare a CSV file similar to dataset_csv/sample_data.csv,
``` shell
python train_mae.py --csv_path ../dataset_csv/sample_data.csv --save_path ../MODELS --data_h5_dir ../tile_result --data_slide_dir ../slides/TCGA-LUNG --data_type tcga_lung
``` 


## Extracting Tile-Level Features
Run data_processing/extract_features_fp.py to extract the tile-level features.
For example, you can use following command for extracting features.  
``` shell
python extract_features_fp.py --data_h5_dir ../tile_results --data_slide_dir ../slides/TCGA-LUNG --csv_path ../dataset_csv/sample_data.csv --feat_dir ../FEATURES --data_type tcga_lung --model_path ../MODELS/adco_tcga_lung_not_sym.pth.tar
```  
The above command will use the trained ADCO model in ```model_path``` to extract tile features in ```data_slide_dir```
and save the features to ```feat_dir```. 

## Training TIME Regression Model
Run train/train_att_TIME.py to perform downstream regression task. For example:  
``` shell
python train_att_TIME.py --feature_path ../FEATURES --train_csv_path xxx.csv --val_csv_path xxx.csv
```
The above command will use the feature file in ```data_root_dir``` to train the regression model, and then output the test results to ```results_dir```.
User needs to divide the data set into training set, verification set and test set in advance and put them under dataset_csv/tumor, such as:  

## Training Classification Model

Run train/train_tumor.py to perform downstream classification task. For example:  
``` shell
python train_tumor.py --data_root_dir ../FEATURES --extract_model ADCO --results_dir ../results
```  
The above command will use the feature file in ```data_root_dir``` to train the classification model, and then output the test results to ```results_dir```.
User needs to divide the data set into training set, verification set and test set in advance and put them under dataset_csv/clf, such as:  
``` bash
dataset_csv/clf
	     ├── train_dataset_1.csv
	     ├── ...
	     ├── train_dataset_5.csv
	     ├── test_dataset_1.csv
	     ├── ...
	     ├── test_dataset_5.csv
	     ├── val_dataset_1.csv
	     ├── ...
	     ├── val_dataset_5.csv
```  
The default number of folds is 5, if user want to change fold numbers, add ```--k fold_number``` and prepare corresponding training files in dataset_csv/clf.
The training files is like dataset_csv/clf/sample_data2.csv.

