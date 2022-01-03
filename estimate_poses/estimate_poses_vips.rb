require "onnxruntime"
require "mini_magick"
require "vips"

input_img = Vips::Image.new_from_file("images/#{ARGV[0]}")
hscale = 192.0 / input_img.width
vscale = 192.0 / input_img.height

p "hscale: #{input_img.width * hscale}"
p "vscale: #{input_img.height * vscale}"

input_resize = input_img.resize(hscale, vscale: vscale)
p input_resize
input_resize.write_to_file('results/resize.jpg')
pixels = input_resize.to_a

# モデルの読み込み
model = OnnxRuntime::Model.new("movenet_singlepose_lightning_4.onnx")
p model.inputs # モデルのインプット形式を確認できる
p model.outputs # モデルのアウトプット形式を確認できる

# 姿勢推定の実行
result = model.predict({'input' => [pixels]})
p result['output_0'][0][0] # 姿勢推定結果

# 確認用なので以下はminimagickのまま
=begin
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
=end