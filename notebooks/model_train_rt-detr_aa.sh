name=multi_state_80_train_20_test
task=aa
suffix=v1
model_dir=/home/rishabh.mondal/Brick-Kilns-project/albk_rishabh/albk_v2/YOLO_LOCALIZATION/
model=rtdetr-x.pt
base_path=/home/rishabh.mondal/Brick-Kilns-project/albk_rishabh/albk_v2/YOLO_LOCALIZATION
data=$base_path/all_data.yaml
imgsz=1120
epochs=100
device=0
# checkpoint=10
experimentName=$name\_$task\_$suffix\_$model\_$imgsz\_$epochs

echo "Name: $name"
echo "Task: $task"
echo "Suffix: $suffix"
echo "Experiment Name: $experimentName"
echo "Model: $model"
echo "Data: $data"
echo "Image Size: $imgsz"
echo "Epochs: $epochs"
echo "Device: $device"


nohup yolo obb train model=$model_dir/$model data=$data imgsz=$imgsz device=$device name=$base_path/runs/$experimentName epochs=$epochs val=False save_conf=True save_txt=True save=True > $base_path/$experimentName.log 2>&1 &
echo "Job fired!"
