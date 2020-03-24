
# coding: utf-8


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

DATA_PATH = '../data/'
LIGHTCURVES_PATH = DATA_PATH + 'lightcurves/'
FEATURES_PATH = DATA_PATH + 'features/'

import numpy as np
import pandas as pd
import measurements
import extract
import inputs2
from multiprocessing import cpu_count, Pool, current_process


def unique_ids_list(df_lcs):
    return df_lcs.index.get_level_values('ID').unique().format()


def print_num_ids_shape(df_lcs):
    unique_ids = unique_ids_list(df_lcs)
    print('Num IDs: {}  Shape: {}'.format(len(unique_ids), df_lcs.shape))
# #### Import

# Import __transient__ catalogue


df_cat = inputs2.load_transient_catalog()
# Import __transient__ lightcurves


filename = 'transient_lightcurves_clean.csv'
indir = LIGHTCURVES_PATH
filepath = indir + filename
df_transient_noclass = pd.read_csv(filepath)
df_transient_noclass = df_transient_noclass.set_index(['ID', 'observation_id'])
df_transient_noclass.head()
print_num_ids_shape(df_transient_noclass)
# Import __non-transient__ light curves


# ids unicos
ids = df_transient_noclass.index.get_level_values('ID').unique()
# escoger aleatoriamente 25% de los indices

testInd = np.random.choice(ids, int(0.25*len(ids)), replace=False)
# sacar dataframes

testdf = df_transient_noclass[df_transient_noclass.index.get_level_values(
    'ID').isin(testInd)]
trainningfd = df_transient_noclass[~df_transient_noclass.index.get_level_values(
    'ID').isin(testInd)]

t = [1, 2, 3]

filename = 'nontransient_lightcurves_clean.csv'
indir = LIGHTCURVES_PATH
filepath = indir + filename
df_nont = pd.read_csv(filepath)

df_nont = df_nont.set_index(['ID', 'observation_id'])
print_num_ids_shape(df_nont)
# #### Add class

# __Transient__


df_tra = df_transient_noclass.join(df_cat, how='inner')

df_tra.head()
# __Non-Transient__


df_nont['class'] = 'non-transient'
# #### Filter


def filter_light_curves(df_lcs, min_obs):
    df_count = df_lcs.groupby('ID', as_index=True).count()
    df_count['ObsCount'] = df_count['Mag']
    df_count = df_count[['ObsCount']]
    df_lcs_with_counts = df_lcs.join(df_count, how='inner')
    # Remove objects with less than min_obs
    df_filtered = df_lcs_with_counts[df_lcs_with_counts.ObsCount >= min_obs]
#     # Remove ObsCount
#     df_filtered = df_filtered.drop(['ObsCount'], axis=1)
    return df_filtered


def sample(df_lcs, num_samples):
    # Set random seed
    np.random.seed(42)
    # Sample non-transient subset of same size as transients
    IDs = np.random.choice(unique_ids_list(
        df_lcs), size=num_samples, replace=False)
#     print(IDs); return
    df_sampled = df_nont.loc[IDs]
    return df_sampled


# Filter __transient__ light curves


df_tra_5 = filter_light_curves(df_tra, 5)
print_num_ids_shape(df_tra_5)

del df_tra


# Filter __non-transient__ lightcurves


df_nont_5 = filter_light_curves(df_nont, 5)
print_num_ids_shape(df_nont_5)
# #### Oversample


def oversample(df_lcs, copies=0):
    df_oversample = df_lcs.copy()
    df_oversample['copy_num'] = 0
    for i in range(1, copies+1):
        df_temp = df_lcs.copy()
        df_temp['copy_num'] = i
        df_temp['Mag'] = np.random.normal(df_lcs.Mag, df_lcs.Magerr)
        df_oversample = df_oversample.append(df_temp)
    df_oversample = df_oversample.set_index(['copy_num'], append=True)
    return df_oversample


# Oversample __transient__ light curves


df_tra_5_os = oversample(df_tra_5, 10)
print_num_ids_shape(df_tra_5_os)

del df_tra_5


# "Oversample" __nontransient__ light curves


df_nont_5_os = oversample(df_nont_5, 0)
print_num_ids_shape(df_nont_5)

del df_nont_5


# #### Feature Extraction


def extract_features(df_lcs):
    pid = (current_process().name.split('-')[1])
    print("Process ", pid, " starting...")
    print("Process ", pid, " extracting num_copy...")
    # Extract num_copy list
    num_copy_list = df_lcs.index.get_level_values('copy_num').unique()
    num_copies = len(num_copy_list)
    print("Process ", pid, " extracting id_list...")
    # Extract IDs list
    unique_ids_list = df_lcs.index.get_level_values('ID').unique()
    num_ids = len(unique_ids_list)
    print("Process ", pid, " creating ouput vars...")
    # Create empty feature dict
    feats_dict = extract.feature_dict(30)
    feats_dict['ObsCount'] = []
    feats_dict['Class'] = []
    # Add 'ID' and 'copy_num' index lists
    index_id_list = []
    index_copy_num_list = []
    print("Process ", pid, " starting processing loop...")
    num_objects = num_ids*num_copies
    for num_copy in num_copy_list:
        for i, obj_id in enumerate(unique_ids_list):
            # Print status
            current_object_i = (num_copy+1)*(i+1)
#             if(current_object_i%int(num_objects/1000) == 0):
            print('Process #:', pid, " ", current_object_i, '/', num_objects)
            # Get current object light curve
            print(pid, current_object_i, 'geting object light curve')
            df_object = df_lcs.loc[obj_id, :, num_copy]
#             print(feats_dict)
#             break
            # Get features
            print(pid, current_object_i, 'extracting features...')
            try:
                obj_feats = extract.features(df_object, feats_dict)
            except Exception as e:
                print(pid, current_object_i, 'Encountered exception:')
                print(pid, current_object_i, e)
                print(pid, current_object_i, "continuing loop...")
                continue
            print(pid, current_object_i, 'features extracted.')
#             print(obj_feats)
#             break
            # Append features
            print(pid, current_object_i, 'appending features')
            for k, v in obj_feats.items():
                feats_dict[k].append(obj_feats[k])
            # Append Indexes
            print(pid, current_object_i, 'appending indices.')
            index_id_list.append(obj_id)
            index_copy_num_list.append(num_copy)
            # Append class and obs_count
            #assert(len(df_object['class'].unique()) == 1)
            #assert(len(df_object['ObsCount'].unique()) == 1)
            #assert(df_object['ObsCount'].unique()[0] == df_object.shape[0])
            feats_dict['Class'].append(df_object['class'].unique()[0])
            feats_dict['ObsCount'].append(df_object.shape[0])
            print(pid, current_object_i, 'done with object')
    print(pid, 'finished processing loop.')
    # Create feature dataframe
    print(pid, 'creating feature dataframe')
    df_feats = pd.DataFrame(feats_dict).set_index(
        [index_id_list, index_copy_num_list])
    df_feats.index.names = ['ID', 'copy_num']
    # NEED TO SAVE A COPY OF DF JUST IN CASE
    print(pid, 'saving temporal copy of dataframe')
    outdir = FEATURES_PATH
    df_feats.to_csv(outdir + str(pid) + ".csv")
    print(pid, 'done with extraction')
    return df_feats


# def save_features(df_feats, obj_type):
#     outdir = FEATURES_PATH
#     filename_raw = '{}.csv'
#     filename = filename_raw.format(obj_type)
# #     assert(df_feats.shape[1]==32) # 30 + ['num_obs'+'class']
#     df_feats.to_csv(outdir + filename)
# #### Generate Features


def extractSalt2Parallel(df_all,name):  # , transient, min_obs):
    # obj_type = 'T' if transient else 'NT'
    # init parallel params
    cores = cpu_count()+2
    pool = Pool(cores)
    # split dataframe into equal parts
    # one for each core
    ids = np.array(df_all.index.get_level_values('ID').unique())
    np.random.shuffle(ids)
    split_ids = np.array_split(ids, cores)
    dfs = [df_all[df_all.index.get_level_values(
        'ID').isin(id_set)] for id_set in split_ids]
    # execute extraction in parallel
    print('Starting parallelization')
    pool.map(extractSALT2ForParallel, dfs)
    print('Done parallelization, closing and joining')
    pool.close()
    pool.join()
    print('Done parallelization, closed and joined')
#     return '--------------'
    # Generate features based on light curves in parallel
    #df_feats = extract_features(df_all,obj_type)
    #spl = np.array_split(data, partitions)
    # print('saving entire df')
    # save_features(df_feats, obj_type)
    # print('saved entire df')
    # Log Finished
    print('Finished task type={} obs={}'.format(obj_type, min_obs))
    return df_feats


def extractSalt2Serial(df_all, name):  # , transient, min_obs):

    # ids = np.array(df_all.index.get_level_values('ID').unique())
    print('saving entire df')
    extractSALT2(df_all, name)
    print('saved entire df')
    # Log Finished
    print('Finished task type={} obs={}'.format(obj_type, min_obs))
    return df_feats



def extractSALT2ForParallel(df_lcs):
    pid = (current_process().name.split('-')[1])

    # Extract num_copy list
    num_copy_list = df_lcs.index.get_level_values('copy_num').unique()
    num_copies = len(num_copy_list)

    # Extract IDs list
    unique_ids_list = df_lcs.index.get_level_values('ID').unique()
    num_ids = len(unique_ids_list)

    # Create empty feature dict
    feats_dict = dict()
    feats_dict['ObsCount'] = []
    feats_dict['Class'] = []
    # Add 'ID' and 'copy_num' index lists
    index_id_list = []
    index_copy_num_list = []

    num_objects = num_ids*num_copies
    for num_copy in num_copy_list:
        for i, obj_id in enumerate(unique_ids_list):
            # Print status
            current_object_i = (num_copy+1)*(i+1)

            print(pid,current_object_i, '/', num_objects)
            # Get current object light curve
            print(pid,current_object_i, 'geting object light curve')
            df_object = df_lcs.loc[obj_id, :, num_copy]

            # Get features
            print(pid,current_object_i, 'extracting Salt2...')
            print(pid, 'obj_id: ', obj_id, "num_copy: ",num_copy)
            # print(df_object)
            try:
                chi2Salt2 = extract.extractSalt2(df_object)
            except Exception as e:
                print(pid,current_object_i, 'Encountered exception:')
                print(pid,current_object_i, e)
                print(pid,current_object_i, "continuing loop...")
                continue
            print(pid,current_object_i, 'salt2 extracted.')

            f = open("./chi2Salt2/"+pid + "_salt2"".dat", "a+")
            f.write(str(obj_id)+" " + str(num_copy) + " " + str(chi2Salt2) +"\n")
            f.close()
    print(pid," DONE EXTRACTING")


def extractSALT2(df_lcs,name):

    # Extract num_copy list
    num_copy_list = df_lcs.index.get_level_values('copy_num').unique()
    num_copies = len(num_copy_list)

    # Extract IDs list
    unique_ids_list = df_lcs.index.get_level_values('ID').unique()
    num_ids = len(unique_ids_list)

    # Create empty feature dict
    feats_dict = dict()
    feats_dict['ObsCount'] = []
    feats_dict['Class'] = []
    # Add 'ID' and 'copy_num' index lists
    index_id_list = []
    index_copy_num_list = []

    num_objects = num_ids*num_copies
    for num_copy in num_copy_list:
        for i, obj_id in enumerate(unique_ids_list):
            # Print status
            current_object_i = (num_copy+1)*(i+1)

            print(current_object_i, '/', num_objects)
            # Get current object light curve
            print(current_object_i, 'geting object light curve')
            df_object = df_lcs.loc[obj_id, :, num_copy]

            # Get features
            print(current_object_i, 'extracting Salt2...')
            # print(df_object)
            try:
                chi2Salt2 = extract.extractSalt2(df_object)
            except Exception as e:
                print(current_object_i, 'Encountered exception:')
                print(current_object_i, e)
                print(current_object_i, "continuing loop...")
                continue
            print(current_object_i, 'salt2 extracted.')

            f = open("salt2"+name+".dat", "a+")
            f.write(str(obj_id)+" " + str(num_copy) + " " + str(chi2Salt2) +"\n")
            f.close()


# Generate features __transient__ light curves

# def extractSalt2(df):
# df_tra_feats = generate_features(df_tra_5_os, transient=True, min_obs=5)

df_tra_salt2 = extractSalt2Parallel(df_tra_5_os, 'transient')

for i in range(10):
    print("-"[0]*100)
print("STARTING NON TRANSIENTS \n"*100)
for i in range(10):
    print("-"[0]*100)
# df_nont_feats = generate_features(df_nont_5_os, transient=False, min_obs=5)

df_nont_salt2 = extractSalt2Parallel(df_nont_5_os, 'nonTransient')