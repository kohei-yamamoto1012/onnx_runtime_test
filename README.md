# 実行手順
## 物体検出 
- detect_objects/imagesに入力画像を配置

- 姿勢推定を実行
```
cd detect_objects/
be ruby detect_objects.rb input.jpeg
```

## 姿勢推定 
- estimate_poses/imagesに入力画像を配置

- 姿勢推定を実行
```
cd estimate_poses/
be ruby estimate_vips.rb input.jpeg
```