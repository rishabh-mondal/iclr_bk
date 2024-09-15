name=multi_state_data_train_80_test_amedabad_10_km_buffer
task=detect
suffix=v1
root_path=/home/rishabh.mondal/Brick-Kilns-project/albk_rishabh/albk_v2/YOLO_LOCALIZATION
base_path=/home/rishabh.mondal/Brick-Kilns-project/albk_rishabh/albk_v2/YOLO_LOCALIZATION/processed_data/amedabad_10_km_buffer_data
state_part_name=validation_yolo_v10x
data=/home/rishabh.mondal/Brick-Kilns-project/albk_rishabh/albk_v2/YOLO_LOCALIZATION/obb_val_sh.yaml
imgsz=1120
epochs=100
device=2
experimentName=$name\_$task\_$suffix\_$model\_$imgsz\_$epochs
model=/home/rishabh.mondal/Brick-Kilns-project/albk_rishabh/albk_v2/YOLO_LOCALIZATION/runs/multi_state_80_train_20_test_aa_v1_rtdetr-x.pt_1120_100/weights/best.pt
log_dir=$root_path/$experimentName/$state_part_name
log_file=$log_dir/$state_part_name.log


echo "Name: $name"
echo "Task: $task"
echo "Suffix: $suffix"
echo "Experiment Name: $experimentName"
# echo "Model: $model"
echo "Data: $data"
echo "Image Size: $imgsz"
echo "Epochs: $epochs"
echo "Device: $device"
mkdir -p $log_dir

nohup rtdetr detect val model=$model data=$data conf=0.25 imgsz=$imgsz iou=0.5 device=$device name=$root_path/validation/$experimentName/$state_part_name save_txt=True save=False save_conf=True save_crop=False verbose=True > $log_file 2>&1 &
echo "Job fired!"
