---
sidebar_position: 43
title: 🏷️ 数据标注
---

# 数据标注

数据标注是机器学习项目中关键但耗时的环节。

## 标注工具

### Label Studio

```bash
# 安装
pip install label-studio

# 启动
label-studio start
```

```python
# API 使用
from label_studio_sdk import Client

ls = Client(url='http://localhost:8080', api_key='xxx')
project = ls.start_project(
    title='图像分类',
    label_config='''
    <View>
        <Image name="image" value="$image"/>
        <Choices name="choice" toName="image">
            <Choice value="Cat"/>
            <Choice value="Dog"/>
        </Choices>
    </View>
    '''
)
```

### CVAT (计算机视觉标注)

```bash
# Docker 部署
docker-compose up -d
```

支持任务：

- 图像分类
- 目标检测
- 语义分割
- 视频标注

### Prodigy

```bash
# 文本分类标注
prodigy textcat.teach my_dataset en_core_web_sm ./data.jsonl

# 主动学习循环
prodigy train ./model --textcat my_dataset
```

## 标注格式

### COCO 格式

```json
{
  "images": [
    { "id": 1, "file_name": "image.jpg", "width": 640, "height": 480 }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 100, 200, 150],
      "area": 30000,
      "iscrowd": 0
    }
  ],
  "categories": [{ "id": 1, "name": "cat" }]
}
```

### YOLO 格式

```
# class_id x_center y_center width height (归一化)
0 0.5 0.5 0.3 0.4
```

### VOC 格式

```xml
<annotation>
    <object>
        <name>cat</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>100</ymin>
            <xmax>300</xmax>
            <ymax>250</ymax>
        </bndbox>
    </object>
</annotation>
```

## 标注质量控制

```python
# 一致性检查
def compute_agreement(annotations_1, annotations_2):
    # Cohen's Kappa
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(annotations_1, annotations_2)

# 黄金标准测试
def quality_check(annotator_labels, gold_labels):
    accuracy = (annotator_labels == gold_labels).mean()
    return accuracy > 0.9
```

## 半自动标注

```python
# 模型预标注 + 人工校正
def pre_annotate(model, images):
    predictions = model.predict(images)
    # 导出到标注工具进行人工校正
    return predictions

# 主动学习选择
def select_for_annotation(model, unlabeled_data, n=100):
    uncertainty = model.predict_proba(unlabeled_data).max(axis=1)
    uncertain_idx = uncertainty.argsort()[:n]
    return unlabeled_data[uncertain_idx]
```

## 工具对比

| 工具         | 类型   | 优点       | 缺点     |
| ------------ | ------ | ---------- | -------- |
| Label Studio | 通用   | 开源、灵活 | 需要部署 |
| CVAT         | CV     | 功能强大   | 配置复杂 |
| Prodigy      | NLP    | 主动学习   | 付费     |
| Labelbox     | 云服务 | 易用       | 成本高   |
