# m6A-IIN

## Requirements

* Python 3.7
* PyTorch = 1.8.0
* numpy = 1.19.5

## Data Preprocess

- The raw data and preprocessing process can be found in the *Pre-Encoding* folder, which contains three subfolders: *data*, *Dataset*, and *Preprocess*:
  - `data`: The raw datasets used by the model.
  - `Dataset`: The processed data, including primary and secondary structure information.
  - `Preprocess`: Contains four Python files. Executing them in sequence will generate the input data for model training and validation.

## data_loader

- `SiteBinding_dataloader1.py`: Execute the file to read the training and testing dataset files.

## models

- Our model can be found in the `models` folder, with brief descriptions of the various Python files as follows:
  - `configure.py`: Configuration file for some settings.
  - `ELMom6A.py`: File for semantic information extraction based on ELMo.
  - `Graph_Wavelet.py`: File for the graph wavelet convolution layer.
  - `GraphInteraction.py`: File related to the coupling layer.
  - `handler.py`: Data processing file.
  - `IBGNN.py`: Integrated model file.
  - `Utils.py`: Utility file.

## Data_analysis

- The folder contains the processing code related to the experiments. Brief descriptions are as follows:
  - `Main_experiments`: Result files of the main experiments in Section 2.2, where the visualization of results is completed using the Omicshare tool (https://www.omicshare.com/tools/).
  - `Primary_and_secondary_structure_analysis`: Some experimental results and analysis codes from Section 2.3, with more detailed results available in the supplementary materials.
    - In `0The_ratio_of_human_datasets`, we present the ratio of different types of information within methylated samples with identified binding regions or loops in the H-b, H-k, and H-l datasets, including samples with solely identified binding regions, samples with overlapping binding regions and loops, and samples with solely identified loop information.
    - In the `binding_region_loop_alternation` folder, we explain the adjustments made to binding regions and loops, and provide visualization methods.
    - `1Visualization_of_the_motifs.py` demonstrates how we identify and visualize motif regions.
  - `The_evolutionary_difference`: Analysis code for t-SNE in Section 2.4, with more detailed results available in the supplementary materials.
  - `Pan-cancer analysis`: Analysis code for histograms and density plots in Section 2.5, with more detailed results available in the supplementary materials.

