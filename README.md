# 使用 RankIQA 的方法来做人脸姿态
## 数据预处理
训练数据: 300W_LP, 数据量122450

验证集: AFLW2000，数据量 2000

* 从 300W_LP 中提取出人脸三维姿态，保存在txt中，保存格式 image_path yaw pitch roll
  更改 extract_pose.py 中的保存路径
  run python extract_pose.py --input_file 300W_LP/AFW 
      python extract_pose.py --input_file 300W_LP/AFW_Flip
      python extract_pose.py --input_file 300W_LP/............ 
* 从 AFLW2000 中提取人脸三维姿态
  更改 extract_pose.py中的保存路径
  run python extract_pose.py --input_file AFLW2000
* 将三维角度分级
  角度范围[-99, 102], 3度一个分级
  更改 rankdata.py 中的 rootpath 与 savepath，分别处理 300W_LP 与 AFLW2000
  run python rankdata.py
* MTCNN 检测扣除人脸
