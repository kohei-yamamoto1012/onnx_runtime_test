require "onnxruntime"
require "mini_magick"
require "vips"
require 'matrix'

INPUT_IMG_WIDTH = 192
INPUT_IMG_HEIGHT = 192

KEYPOINT_INDEX = {
  nose: 0,
  left_eye: 1,
  right_eye: 2,
  left_ear: 3,
  right_ear: 4,
  left_shoulder: 5,
  right_shoulder: 6,
  left_elbow: 7,
  right_elbow: 8,
  left_wrist: 9,
  right_wrist: 10,
  left_hip: 11,
  right_hip: 12,
  left_knee: 13,
  right_knee: 14,
  left_ankle: 15,
  right_ankle: 16
}

def draw_result_line(img, keypoint1, keypoint2)
  img.draw_line(
    [255, 0, 0],
    keypoint1[:x],
    keypoint1[:y],
    keypoint2[:x],
    keypoint2[:y],
  )
end

def calc_angle(vertex_keypoint, side_a_keypoint, side_b_keypoint)
  a1 = vertex_keypoint[:x] - side_a_keypoint[:x]
  a2 = vertex_keypoint[:y] - side_a_keypoint[:y]
  vector_a = Vector[a1, a2]

  b1 = vertex_keypoint[:x] - side_b_keypoint[:x]
  b2 = vertex_keypoint[:y] - side_b_keypoint[:y]
  vector_b = Vector[b1, b2]

  angle = (vector_a.angle_with(vector_b) * 180 / Math::PI).round
end

# 入力画像の加工
input_img = Vips::Image.new_from_file("images/#{ARGV[0]}").autorot
input_img = input_img[0..2] if input_img.bands > 3
resize = input_img.thumbnail_image(INPUT_IMG_WIDTH, height: INPUT_IMG_HEIGHT) # アスペクト比を保持しつつ変換
padding = resize.embed(0, 0, INPUT_IMG_WIDTH, INPUT_IMG_HEIGHT, extend: :black) # 0埋め
padding.jpegsave("results/padding.jpeg")

# 0埋めしたピクセル幅・高さを求めておく(resize前基準で)
x_pad = 0
y_pad = 0
x_pad = input_img.height - input_img.width if input_img.height > input_img.width
y_pad = input_img.width - input_img.height if input_img.width > input_img.height

pixels = padding.to_a

# モデルの読み込み
model = OnnxRuntime::Model.new("movenet_singlepose_lightning_4.onnx")
# p model.inputs # モデルのインプット形式を確認できる
# p model.outputs # モデルのアウトプット形式を確認できる

# 姿勢推定の実行
result = model.predict({'input' => [pixels]})

# 画像に検出結果を書き込む
output_img = Vips::Image.new_from_file("images/#{ARGV[0]}").autorot
output_img = output_img[0..2] if output_img.bands > 3

# 姿勢推定結果を入力画像のpx基準の値に変換
result_px = []
result['output_0'][0][0].each do |keypoint|
  x = (keypoint[1] * (output_img.width + x_pad)).round
  y = (keypoint[0] * (output_img.height + y_pad)).round
  score = (keypoint[2] * 100).round
  result_px.push({x: x, y: y, score: score})
end

# 検出結果の関節点を書き込む
result_px.each_with_index do |keypoint, i|
  if i % 2 == 0 
    color = [0, 0, 255] # 右側の関節 = 青色
  else
    color = [255, 0, 0] # 左側の関節 = 赤色
  end
  radius = 10
  output_img = output_img.draw_circle(color, keypoint[:x], keypoint[:y], radius, fill: true)
end

# 検出結果の骨格線を書き込む
output_img = draw_result_line(output_img, result_px[KEYPOINT_INDEX[:left_shoulder]], result_px[KEYPOINT_INDEX[:right_shoulder]])
output_img = draw_result_line(output_img, result_px[KEYPOINT_INDEX[:left_shoulder]], result_px[KEYPOINT_INDEX[:left_elbow]])
output_img = draw_result_line(output_img, result_px[KEYPOINT_INDEX[:left_elbow]], result_px[KEYPOINT_INDEX[:left_wrist]])
output_img = draw_result_line(output_img, result_px[KEYPOINT_INDEX[:left_shoulder]], result_px[KEYPOINT_INDEX[:left_hip]])
output_img = draw_result_line(output_img, result_px[KEYPOINT_INDEX[:left_hip]], result_px[KEYPOINT_INDEX[:left_knee]])
output_img = draw_result_line(output_img, result_px[KEYPOINT_INDEX[:left_knee]], result_px[KEYPOINT_INDEX[:left_ankle]])
output_img = draw_result_line(output_img, result_px[KEYPOINT_INDEX[:right_shoulder]], result_px[KEYPOINT_INDEX[:right_elbow]])
output_img = draw_result_line(output_img, result_px[KEYPOINT_INDEX[:right_elbow]], result_px[KEYPOINT_INDEX[:right_wrist]])
output_img = draw_result_line(output_img, result_px[KEYPOINT_INDEX[:right_shoulder]], result_px[KEYPOINT_INDEX[:right_hip]])
output_img = draw_result_line(output_img, result_px[KEYPOINT_INDEX[:right_hip]], result_px[KEYPOINT_INDEX[:right_knee]])
output_img = draw_result_line(output_img, result_px[KEYPOINT_INDEX[:right_knee]], result_px[KEYPOINT_INDEX[:right_ankle]])
output_img = draw_result_line(output_img, result_px[KEYPOINT_INDEX[:right_hip]], result_px[KEYPOINT_INDEX[:left_hip]])

# 各関節の角度を標準出力
puts "右肩: #{calc_angle(result_px[KEYPOINT_INDEX[:right_shoulder]], result_px[KEYPOINT_INDEX[:right_elbow]], result_px[KEYPOINT_INDEX[:left_shoulder]])}"
puts "左肩: #{calc_angle(result_px[KEYPOINT_INDEX[:left_shoulder]], result_px[KEYPOINT_INDEX[:left_elbow]], result_px[KEYPOINT_INDEX[:right_shoulder]])}"
puts "右肘: #{calc_angle(result_px[KEYPOINT_INDEX[:right_elbow]], result_px[KEYPOINT_INDEX[:right_wrist]], result_px[KEYPOINT_INDEX[:right_shoulder]])}"
puts "左肘: #{calc_angle(result_px[KEYPOINT_INDEX[:left_elbow]], result_px[KEYPOINT_INDEX[:left_wrist]], result_px[KEYPOINT_INDEX[:left_shoulder]])}"
puts "右足: #{calc_angle(result_px[KEYPOINT_INDEX[:right_hip]], result_px[KEYPOINT_INDEX[:right_knee]], result_px[KEYPOINT_INDEX[:left_hip]])}"
puts "左足: #{calc_angle(result_px[KEYPOINT_INDEX[:left_hip]], result_px[KEYPOINT_INDEX[:left_knee]], result_px[KEYPOINT_INDEX[:right_hip]])}"
puts "右膝: #{calc_angle(result_px[KEYPOINT_INDEX[:right_knee]], result_px[KEYPOINT_INDEX[:right_hip]], result_px[KEYPOINT_INDEX[:right_ankle]])}"
puts "左膝: #{calc_angle(result_px[KEYPOINT_INDEX[:left_knee]], result_px[KEYPOINT_INDEX[:left_hip]], result_px[KEYPOINT_INDEX[:left_ankle]])}"


# 画像の出力
output_img.jpegsave("results/#{ARGV[0]}")
