from src.models.cv import (
    ResNet18,
    MobileNetV2,
    EfficientNetB0,
    UNet,
    DeepLabV3,
    FasterRCNN,
    SSD,
    YOLOv3,
)
from src.datasets.cv import (
    get_cifar10_dataloader,
    get_cifar100_dataloader,
    get_imagenet_dataloader,
    get_ade20k_dataloader,
    get_cityscapes_dataloader,
    get_coco_dataloader,
    get_voc_dataloader,
)
from src.optimizers.sgd import optimize_with_sgd
from src.optimizers.adam import optimize_with_adam
from src.optimizers.pso import optimize_with_pso
from src.optimizers.cma_es import optimize_with_cma
from src.optimizers.sa import optimize_with_sa

def run_all_cv_experiments():
    classification_models = [
        ResNet18(num_classes=10),
        MobileNetV2(num_classes=10),
        EfficientNetB0(num_classes=10),
    ]
    classification_dataloaders = [
        get_cifar10_dataloader(batch_size=32),
        get_cifar100_dataloader(batch_size=32),
        get_imagenet_dataloader(batch_size=32),
    ]

    segmentation_models = [
        UNet(num_classes=2),
        DeepLabV3(num_classes=2),
    ]
    segmentation_dataloaders = [
        get_ade20k_dataloader(batch_size=16),
        get_cityscapes_dataloader(batch_size=16),
    ]

    detection_models = [
        FasterRCNN(num_classes=5),
        SSD(num_classes=5),
        YOLOv3(num_classes=5),
    ]
    detection_dataloaders = [
        get_coco_dataloader(batch_size=8),
        get_voc_dataloader(batch_size=8),
    ]

    optimizers = [
        optimize_with_sa,
        optimize_with_cma,
        optimize_with_pso,
        optimize_with_sgd,
        optimize_with_adam,
    ]

    # Classification
    for model in classification_models:
        for dataloader in classification_dataloaders:
            for opt_fn in optimizers:
                print(
                    f"[Classification] {model.__class__.__name__}, "
                    f"{opt_fn.__name__}, "
                    f"{type(dataloader.dataset).__name__}"
                )
                opt_fn(model, dataloader, task_type="classification")

    # Segmentation
    for model in segmentation_models:
        for dataloader in segmentation_dataloaders:
            for opt_fn in optimizers:
                print(
                    f"[Segmentation] {model.__class__.__name__}, "
                    f"{opt_fn.__name__}, "
                    f"{type(dataloader.dataset).__name__}"
                )
                opt_fn(model, dataloader, task_type="segmentation")

    # Detection
    for model in detection_models:
        for dataloader in detection_dataloaders:
            for opt_fn in optimizers:
                print(
                    f"[Detection] {model.__class__.__name__}, "
                    f"{opt_fn.__name__}, "
                    f"{type(dataloader.dataset).__name__}"
                )
                opt_fn(model, dataloader, task_type="detection")

if __name__ == "__main__":
    run_all_cv_experiments()