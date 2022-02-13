# Image captioning
 End-to-end image captioning with **ResNet50 + LSTM with Attention**
 
 Model is seq2seq model. 
 In the encoder pretrained ResNet50 model is used to extract the features. 
 Decoder is the LSTM with the Bahdanau Attention. 
 
# Dataset
The dataset is available at [kaggle](https://www.kaggle.com/adityajn105/flickr8k) and contains 8,000 images that are each paired with five different captions.

# Usage
run in terminal: `python -m img_caption`

# Config 
The user interface consists of file:

* [config.yaml](https://github.com/YuzhaninaAnya/image-captioning/blob/main/img_caption/config.yaml) - general configuration with data and model parameters

Default **config.yaml**: 
````
data:
  path_to_data_folder: "data"
  caption_file_name: "captions.txt"
  images_folder_name: "Images"
  output_folder_name: "output"
  logging_file_name: "logging.txt"
  model_file_name: "model.pt"

batch_size: 32
num_worker: 1
gensim_model_name: "glove-wiki-gigaword-200"

model:
  embedding_dimension: 200
  decoder_hidden_dimension: 300
  learning_rate: 0.0001
  momentum: 0.9
  n_epochs: 50
  clip: 5
  fine_tune_encoder: false
````

# Output 
After training the model, the pipeline will return the following files:
* `model.pt` - checkpoint with: 
    * `epoch` - last epoch 
    * `model_state_dict` - model parameters
    * `optimizer_state_dict` - the state of the optimizer
    * `train_history` - training history from a model
    * `valid_history` - validation history from a model
    * `best_valid_loss` - the best validation loss
