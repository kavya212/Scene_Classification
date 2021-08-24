import splitfolders
input_folder = "Input_Dataset"
output ="Processed_Data2"
splitfolders.ratio(input_folder, output, seed=42, ratio=(.6, .2, .2))

