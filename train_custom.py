import os
from ultralytics import YOLO
import comet_ml #logging hyprparameters
def main():

    # ---- Comet Experiment ----
    # experiment = Experiment(
    #     api_key=os.getenv("COMET_API_KEY"),   # safer than hardcoding
    #     project_name="doclaynet-yolo",
    #     workspace="your_workspace_name",
    #     auto_output_logging="simple",
    # )

    # ---- model ----
    model = YOLO("yolo26n.yaml")

    # ---- log hyperparameters ----
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

    # ---- training ----
    results = model.train(
        data="doclaynet.yaml",
        epochs=50,
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

        patience=20,
        cos_lr=True,
        amp=True,
        cache=False,  
        # workers=4,    # Avoid multiprocessing crashes

        project="/run/media/akshajtiwari/5438E16938E14B16/yolo26_doclaynet",
        name="exp1",
        save=True,
        save_period=2,
        resume=False
    )

    # experiment.end()

comet_ml.init()
if __name__ == "__main__":
    main()