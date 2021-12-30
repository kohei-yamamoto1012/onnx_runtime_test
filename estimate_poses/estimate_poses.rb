require "onnxruntime"
require "mini_magick" # 画像処理用。処理が遅いので変更を検討

# 画像の読み込みとリサイズ
input_img = MiniMagick::Image.open("images/#{ARGV[0]}")
input_img.resize '192x192!'
input_pixels = input_img.get_pixels

# モデルの読み込み
model = OnnxRuntime::Model.new("movenet_singlepose_lightning_4.onnx")
p model.inputs # モデルのインプット形式を確認できる
p model.outputs # モデルのアウトプット形式を確認できる

# 姿勢推定の実行
result = model.predict({'input' => [input_pixels]})
p result['output_0'][0][0] # 姿勢推定結果

# 画像に各keypointの位置を書き込んで出力する
output_img = MiniMagick::Image.open("images/#{ARGV[0]}")
result['output_0'][0][0].each do |keypoint|
  
  # 姿勢推定結果を整数化する
  y = (keypoint[0] * output_img.height).round
  x = (keypoint[1] * output_img.width).round
  score = (keypoint[2] * 100).round

  # keypointの位置に四角形を書き込む
  rectangle_side_half = 10
  left = x - rectangle_side_half
  right = x + rectangle_side_half
  top = y - rectangle_side_half
  bottom = y + rectangle_side_half

  output_img.combine_options do |c|
    c.draw "rectangle #{left},#{top} #{right},#{bottom}"
    c.fill "red"
    c.stroke "red"
  end
end

# 画像の出力
output_img.write("results/#{ARGV[0]}")