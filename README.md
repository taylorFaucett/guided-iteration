# Guided Iteration Example

This is an example of the type of iterative loop one can write for selecting high-level features according to their average decision ordering value. In the class `GuidedIterator`, I have written a method for looping through each feature in a dataset of EFPs (`data/EFP.parquet`) and selecting the best performing example by ADO. This feature is then included as a new feature in model training and the ADO comparison is made again with the remaining EFPs. 

In this example, an XGBoost model (XGBClassifier) is the model trained in the `train_nn` function. However, this can be replaced with any binary classifier. 

The results of each iteration of the loop are saved in the `results` directory. There is an overall results file (`results.csv`) which includes the selected EFP and the AUC and ADO value for the HL+EFP network trained. Additionally, there is a directory for each iteration (e.g. `iteration_0`, `iteration_1`, etc) with the raw predictions, difference ordering values and sorted ado scores for each EFP tested. 

# Example Run

An example run is given in `results` labelled `demo`. This was generated with the demo jupyter notebook `guided_iteration_example.ipynb`. Note that, by default, my code skips steps that have already been completed. So if you want to try and run the entire thing from scratch, you should delete the `demo` directory before running. The final result for this demo data should look something like this:

![Results Demo](https://github.com/taylorFaucett/guided-iteration/blob/main/results/demo/results.png)
