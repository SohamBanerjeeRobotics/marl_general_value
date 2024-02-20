import pandas as pd


dataset = pd.read_csv("../data/randwalk_img2/robomaster_0/current_state/pos.csv")
rel_time = ((dataset['t_rec'] - dataset['t_rec'][0]) / 1e9)
breakpoint()