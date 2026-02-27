# YOLO26n trained from scratch on DocLayNet

import os
from ultralytics import YOLO
import comet_ml #logging hyprparameters
def main():
    model = YOLO("yolo26n.yaml")
    hyperparams = {
        "epochs": 50,
        "imgsz": 640,
        "optimizer": "MuSGD",
        "lr0": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0004,
        "mosaic": 0.3,
        "mixup": 0.05,
        "scale": 0.1,
    }

    # experiment.log_parameters(hyperparams)
    results = model.train(
        data="doclaynet.yaml",
        epochs=100,
        imgsz=640,
        batch=-1,
        device=0,

        optimizer="MuSGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0004,

        hsv_h=0.0,
        hsv_s=0.1,
        hsv_v=0.05,
        degrees=1.5,
        translate=0.02,
        scale=0.1,
        shear=0.3,
        perspective=0.00005,

        mosaic=0.3,
        mixup=0.05,
        erasing=0.1,
        auto_augment="autoaugment",
        close_mosaic=20,

        patience=20, #EARLY STOPPING
        cos_lr=True,
        amp=True,
        cache=False,  
        # workers=4,    # Avoid multiprocessing crashes

        project="/run/media/akshajtiwari/5438E16938E14B16/yolo26_doclaynet",
        name="M1_scratch_real_aug",
        save=True,
        save_period=2,
        resume=False
    )

    # experiment.end()

comet_ml.login()
if __name__ == "__main__":
    main()