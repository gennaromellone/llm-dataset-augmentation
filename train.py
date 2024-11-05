from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback

data = "yolov8/glitter_tx2img_aug/data.yaml"
name = 'GLitter_Txt2Img_model_1gpu'
epochs = 150
batch = 128
lr = 0.0005

wandb.login(key="MY-KEY")

wandb.init(
    project="glitter-paper",
    name='glitter_stablediff-0005',
    config={
    "learning_rate": lr,
    "epochs": epochs,
    "batch_size": batch
    }
)
model = YOLO("yolov8m.pt")

# Add W&B Callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)

# Model training
model.train(
    data=data,
    name=name,
    imgsz=640,
    epochs=epochs,
    batch=batch,
    device="0,1,2,3",
    lr0=lr,
    optimizer='auto'
)

# Finalize the W&B Run
wandb.finish()
