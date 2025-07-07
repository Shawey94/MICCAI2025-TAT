
#! Shawey @ UTA                                                                        
#CN: 0; MCI: 1; AD: 2; LBD: 3                                                              

import os
import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ad_path = '/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/2025Projects/MICCAI2025/ADNI_SC'
ad_info = '/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/2025Projects/MICCAI2025/ADNI_Gender_Age_Score_485.csv'

lbd_path = '/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/2025Projects/MICCAI2025/sc_roi_des_voxel_2_raw/roi_sc_des_raw'
lbd_info = '/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/2025Projects/MICCAI2025/lbd_t1/combined_metadata_lbd_all.csv'


info_df = pd.read_csv(ad_info)
sub_ids = info_df['StudentID'].tolist()
sub_labels = info_df['Research Group'].tolist()

# Prepare file writing
lables_cnt = {'LBD':0, 'CN':0, 'AD':0, 'MCI':0}
unique_lables = []


for img_file in os.listdir(ad_path):
    img_id = os.path.splitext(img_file)[0]  # Remove extension
    if img_id in sub_ids:
        label = sub_labels[sub_ids.index(img_id)]
        if(label == 'EMCI' or label == 'LMCI'):
                    label = 'MCI'
        if(label == 'SMC' ):
                    continue
        if(label == 'Patient' ):
                    continue
        if label not in unique_lables:
                    unique_lables.append(label)
        lables_cnt[label] += 1

        #CN: 0; MCI: 1; AD: 2; LBD: 3
        if(label == 'CN'):
                    label = 0
        elif(label == 'MCI'):
                    label = 1
        elif(label == 'AD'):
                    label = 2
        elif(label == 'LBD'):
                    label = 3

        sc = np.load(ad_path + '/' + img_id+'/'+'common_fiber_matrix.npy') 

        # Check for all-zero rows
        has_zero_row = np.any(np.all(sc == 0, axis=1))

        # Check for all-zero columns
        has_zero_column = np.any(np.all(sc == 0, axis=0))

        # Remove all-zero rows
        sc = sc[~np.all(sc == 0, axis=1)]

        # Remove all-zero columns
        sc = sc[:, ~np.all(sc == 0, axis=0)]

        if sc.shape != (148, 148):
            continue

        # SC preprocessing follow: Predicting brain structural network using functional connectivity
        sc_log = np.log2(sc + 1)

        # Compute mean (μ) and standard deviation (σ)
        u = np.mean(sc_log)
        sigma = np.std(sc_log)

        # Normalize: S = (S - μ) / σ
        sc_normalized = (sc_log - u) / sigma
        ###############################################################################################
        
   
        # Plot heatmap
        plt.figure(figsize=(8, 8))
        sns.heatmap(sc_normalized, cmap="viridis", xticklabels=False, yticklabels=False)
        plt.title("ADNI SC Heatmap Visualization")
        plt.show()
        



# Load CSV file
info_df = pd.read_csv(lbd_info)
sub_ids = info_df['subject_id'].tolist()
sub_labels = info_df['diagnosis_label'].tolist()

# Prepare file writing
lables_cnt = {'LBD':0, 'CN':0, 'AD':0, 'MCI':0}

unique_lables = []

for img_file in os.listdir(lbd_path):
    img_id = os.path.splitext(img_file)[0]                               
    if img_id in sub_ids:
        label = sub_labels[sub_ids.index(img_id)]
        if(label == 'PDD' or label == 'DLB'):
                    label = 'LBD'
        if(label == 'MCI_AD'):
                    label = 'MCI'
        if(label not in ['PDD','CN', 'AD', 'DLB', 'LBD', 'MCI']):
                    continue
        if label not in unique_lables:
                    unique_lables.append(label)
        lables_cnt[label] += 1

        #CN: 0; MCI: 1; AD: 2; LBD: 3
        if(label == 'CN'):
                    label = 0
        elif(label == 'MCI'):
                    label = 1
        elif(label == 'AD'):
                    label = 2
        elif(label == 'LBD'):
                    label = 3

        sc = np.loadtxt(lbd_path + '/' + img_id + '.txt')

        # Check for all-zero rows
        has_zero_row = np.any(np.all(sc == 0, axis=1))

        # Check for all-zero columns
        has_zero_column = np.any(np.all(sc == 0, axis=0))

        # Remove all-zero rows
        sc = sc[~np.all(sc == 0, axis=1)]

        # Remove all-zero columns
        sc = sc[:, ~np.all(sc == 0, axis=0)]

        # SC preprocessing follow: Predicting brain structural network using functional connectivity
        sc_log = np.log2(sc + 1)

        # Compute mean (μ) and standard deviation (σ)
        u = np.mean(sc_log)
        sigma = np.std(sc_log)

        # Normalize: S = (S - μ) / σ
        sc_normalized = (sc_log - u) / sigma
        ###############################################################################################


        # Plot heatmap
        plt.figure(figsize=(8, 8))
        sns.heatmap(sc_normalized, cmap="viridis", xticklabels=False, yticklabels=False)
        plt.title("LBD SC Heatmap Visualization")
        plt.show()
        
        

