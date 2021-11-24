# Usage:
## step1
preprocessing.py將dicom內的image獨立出來<br>
並額外建立一個image資料夾儲存個別的資料
  
## step2
分別使用cut.py, gray_to_rgb.py, he.py對image內的影像進行個別種類的前處理<br>

## step3
使用train.py對前處理完的image 進行train model並且儲存model state<br>

## step4
使用predict.py訓練好的model state對valid data進行分類<br>
