import os
import shutil


def populate_kfold_directories(data_dir, K_FOLDS):

    alarmed_images = os.listdir(f"{data_dir}/Alarmed")
    annoyed_images = os.listdir(f"{data_dir}/Annoyed")
    curious_images = os.listdir(f"{data_dir}/Curious")
    relaxed_images = os.listdir(f"{data_dir}/Relaxed")

    for i in range(K_FOLDS):
        validation_range = (i*20, i*20 + 20)

        for j in range(0, 100):
            if validation_range[0] <= j < validation_range[1]:
                shutil.copy(f"{data_dir}/Alarmed/{alarmed_images[j]}", f"folds/fold{i}/validation/Alarmed/")
                shutil.copy(f"{data_dir}/Annoyed/{annoyed_images[j]}", f"folds/fold{i}/validation/Annoyed/")
                shutil.copy(f"{data_dir}/Curious/{curious_images[j]}", f"folds/fold{i}/validation/Curious/")
                shutil.copy(f"{data_dir}/Relaxed/{relaxed_images[j]}", f"folds/fold{i}/validation/Relaxed/")
            else:
                shutil.copy(f"{data_dir}/Alarmed/{alarmed_images[j]}", f"folds/fold{i}/train/Alarmed/")
                shutil.copy(f"{data_dir}/Annoyed/{annoyed_images[j]}", f"folds/fold{i}/train/Annoyed/")
                shutil.copy(f"{data_dir}/Curious/{curious_images[j]}", f"folds/fold{i}/train/Curious/")
                shutil.copy(f"{data_dir}/Relaxed/{relaxed_images[j]}", f"folds/fold{i}/train/Relaxed/")


def create_kfold_directories(K_FOLDS):

    try:
        os.mkdir("folds")
    except:
        print("Directory 'folds' already exists")

    for i in range(K_FOLDS):
        try:
            os.mkdir(f"folds/fold{i}/")
            os.mkdir(f"folds/fold{i}/train")
            os.mkdir(f"folds/fold{i}/validation")
            os.mkdir(f"folds/fold{i}/train/Alarmed")
            os.mkdir(f"folds/fold{i}/train/Annoyed")
            os.mkdir(f"folds/fold{i}/train/Curious")
            os.mkdir(f"folds/fold{i}/train/Relaxed")
            os.mkdir(f"folds/fold{i}/validation/Alarmed")
            os.mkdir(f"folds/fold{i}/validation/Annoyed")
            os.mkdir(f"folds/fold{i}/validation/Curious")
            os.mkdir(f"folds/fold{i}/validation/Relaxed")
        except:
            print("Can't create directory because it already exists")
