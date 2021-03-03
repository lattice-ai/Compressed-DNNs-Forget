"""Project Config in JSON"""

CFG = {
    "data": {
        "data_folder" : "data/CelebA/",
        "images_folder" : "data/CelebA/img_align_celeba/img_align_celeba/",
        "IMG_HEIGHT": "218",
        "IMG_WIDTH": "178",
        "TRAINING_SAMPLES": "10000"
    },
    "model": {
        "input": [218, 178, 3]
    },
    "train": {
        "batch_size": "64"
    }   
}