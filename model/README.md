Model architectures are created in PyTorch Lightning and modules and dataset class are stored in src/modules.py. 

Parameter sweeps and predictions were created in Google Colab. To run the following colab notebooks, reader will need to mount their Google Drive and access the data and utilities stored in the following [Drive](https://drive.google.com/drive/folders/1b9qMhMblYRnJHzZOqJeFnltUX9jpfLTo?usp=sharing). 

Both the final model and the updated/finetuned model can be accessed from here within the [Drive](https://drive.google.com/drive/folders/1sSKn6YagtzL70m8Ck3N3yppETwmd0hae?usp=sharing).

Alternatively predictions can be made with these models using the following [Colab Notebook](https://colab.research.google.com/drive/1EQUWDyDrzC-ZCKZ-z6f0sTiqZ1TF0BoX?usp=sharing). If running not on Colab, download the Drive folder linked above and use the script Predictions.ipynb. Base directory variable will need to be updated to the location of the HurricaneDamage_Data folder. 

Patches will be logged on wandb both for test and validation images automatically using a wandb image logger. 

To train or finetune the Final model on your own data use the following [Colab Notebook](https://colab.research.google.com/drive/1M_XMjO6K1uJeCehBkYy9ijEa7eX9qDtb?usp=sharing). 

Your data will need to be in the folder structure, with each patch labelled with a numeric identifier e.g. 1234.tif. The datamodules provided will then set up a 80/20 split of the train data into train/validation.

you_datadir:
- train/damaged_Y
- train/damaged_N
- test/damaged_Y
- test/damaged_N
