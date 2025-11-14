import pandas as pd
import matplotlib.pyplot as plt

def cumulative_comparison_plots(
    original_eq_file, declustring_eq_file,
    save_path=None    
):

# データの読み取り
df_org = pd.read_csv(original_eq_file)
df_declus = pd.csv(declustring_eq_file)




if __name__ == "__main__":