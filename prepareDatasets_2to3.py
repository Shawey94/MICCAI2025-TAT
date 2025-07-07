
#! Shawey @ UTA                                                                        
#CN: 0; MCI: 1; LBD: 2                                                           

import os
import pandas as pd                 
import nibabel as nib               
import numpy as np                
 
ad_path = '/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/2025Projects/MICCAI2025/ADNI_SC'
ad_info = '/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/2025Projects/MICCAI2025/ADNI_Gender_Age_Score_485.csv'

lbd_path = '/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/2025Projects/MICCAI2025/sc_roi_des_voxel_2_raw/roi_sc_des_raw'
lbd_info = '/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/2025Projects/MICCAI2025/lbd_t1/combined_metadata_lbd_all.csv'


#########################################################################################################################
# Prepare file writing
# Load CSV file
info_df = pd.read_csv(ad_info)
sub_ids = info_df['StudentID'].tolist()
sub_labels = info_df['Research Group'].tolist()

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


print('adni lables_cnt:', (lables_cnt))


# Load CSV file
info_df = pd.read_csv(lbd_info)
sub_ids = info_df['subject_id'].tolist()
sub_labels = info_df['diagnosis_label'].tolist()

lables_cnt = {'LBD':0, 'CN':0, 'AD':0, 'MCI':0}

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

print('lbd lables_cnt:', (lables_cnt))
##########################################################################################################################


# for adni, the size of SC is 150 * 150, we need to remove the all zeros to reduce size to 148 * 148
if(0):

    out_file = './data/ADLBD_2to3/'+'adni_list_sc_noRemoveZeros.txt'
    if os.path.exists(out_file):                                                                                             
            os.remove(out_file)                                                     
    # Load CSV file
    info_df = pd.read_csv(ad_info)
    sub_ids = info_df['StudentID'].tolist()
    sub_labels = info_df['Research Group'].tolist()

    # Prepare file writing
    lables_cnt = {'LBD':0, 'CN':0, 'AD':0, 'MCI':0}
    unique_lables = []

    additional_zeros_subs = []

    with open(out_file, 'w') as f:
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
                if(label == 'AD' ):
                        continue
                if label not in unique_lables:
                        unique_lables.append(label)
                lables_cnt[label] += 1

                #CN: 0; MCI: 1; AD: 2; LBD: 3
                if(label == 'CN'):
                        label = 0
                elif(label == 'MCI'):
                        label = 1
                elif(label == 'LBD'):
                        label = 2

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
                    additional_zeros_subs.append(img_id)
                    continue

                # SC preprocessing follow: Predicting brain structural network using functional connectivity
                sc_log = np.log2(sc + 1)

                # Compute mean (μ) and standard deviation (σ)
                u = np.mean(sc_log)
                sigma = np.std(sc_log)

                # Normalize: S = (S - μ) / σ
                sc_normalized = (sc_log - u) / sigma
                ###############################################################################################

                f.write(f"{os.path.join(ad_path, img_id+'/'+'common_fiber_matrix.npy')} {label}\n")

    adni_SCs = open(out_file).readlines()
    print('adni sc length:', len(adni_SCs))
    print(f"unique_lables: {unique_lables}")
    print(f"File saved: {out_file}")
    print('adni lables_cnt:', (lables_cnt))
    print('additional zeros subjects: {}, lenght {}'.format(additional_zeros_subs, len(additional_zeros_subs)) )


    lbd_file = './data/ADLBD_2to3/'+'lbd_list_sc_noRemoveZeros_2to3_2voxels.txt'
    if os.path.exists(lbd_file):
            os.remove(lbd_file)

    # Load CSV file
    info_df = pd.read_csv(lbd_info)
    sub_ids = info_df['subject_id'].tolist()
    sub_labels = info_df['diagnosis_label'].tolist()

    # Prepare file writing
    lables_cnt = {'LBD':0, 'CN':0, 'AD':0, 'MCI':0}
    unique_lables = []
    with open(lbd_file, 'w') as f:
        for img_file in os.listdir(lbd_path):
            img_id = os.path.splitext(img_file)[0]                               
            if img_id in sub_ids:
                label = sub_labels[sub_ids.index(img_id)]
                if(label == 'PDD' or label == 'DLB'):
                        label = 'LBD'
                if(label == 'MCI_AD'):
                        label = 'MCI'
                if(label not in ['PDD','CN', 'DLB', 'LBD', 'MCI']):
                        continue
                if label not in unique_lables:
                        unique_lables.append(label)
                lables_cnt[label] += 1

                #CN: 0; MCI: 1; AD: 2; LBD: 3
                if(label == 'CN'):
                        label = 0
                elif(label == 'MCI'):
                        label = 1
                # elif(label == 'AD'):
                #         label = 2
                elif(label == 'LBD'):
                        label = 2

                sc = np.loadtxt(lbd_path + '/' + img_id + '.txt')

                # Check for all-zero rows
                # has_zero_row = np.any(np.all(sc == 0, axis=1))

                # # Check for all-zero columns
                # has_zero_column = np.any(np.all(sc == 0, axis=0))

                # # Remove all-zero rows
                # sc = sc[~np.all(sc == 0, axis=1)]

                # # Remove all-zero columns
                # sc = sc[:, ~np.all(sc == 0, axis=0)]
                
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
                
                f.write(f"{os.path.join(lbd_path, img_id+'.txt')} {label}\n")

    lbd_SCs = open(lbd_file).readlines()
    print('lbd SC length:', len(lbd_SCs))
    print('lables_cnt:', (lables_cnt))
    print(f"unique_lables: {unique_lables}")
    print(f"File saved: {lbd_file}")
