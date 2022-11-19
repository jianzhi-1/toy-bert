# Toy BERT
This is a toy version of BERT, created for UC Berkeley CS182 (Fall 2022) Final Project.

We want to achieve three goals:
- construction of the BERT model and training process
- visualize hidden states
- train a classification task using the <NSP> token

### Set-up
If you are working on Colab, create an empty folder data under the root directory and upload all the files in the 'data' folder to it.

### Data
The data are lyrics from various songs. See under the 'data' folder.
  
TODO: add random shuffing to the data, then try for more epochs.
TODO: may want to collapse the other classes into a helper file instead, otherwise currently it is clogging up the visualization.
TODO: remove all the commas and lower case all the tokens when processing data
TODO: ablation analysis
TODO: add image for the architecture of Toy BERT
TODO: add steps just like HW9

References: https://theaisummer.com/jax-transformer/
