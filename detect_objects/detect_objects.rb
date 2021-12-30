require "onnxruntime"
require "mini_magick"

img = MiniMagick::Image.open("images/#{ARGV[0]}")
pixels = img.get_pixels

model = OnnxRuntime::Model.new("model.onnx")
result = model.predict({"inputs" => [pixels]})

p result["num_detections"]
p result["detection_classes"]

coco_labels = {
  1 => "person",
  2 => "bicycle",
  3 => "car",
  4 => "motorcycle",
  5 => "airplane",
  6 => "bus",
  7 => "train",
  8 => "truck",
  9 => "boat",
  10 => "traffic",
  11 => "fire hydrant",
  12 => "street sign",
  13 => "stop sign",
  14 => "parking meter",
  15 => "bench",
  16 => "bird",
  17 => "cat",
  18 => "dog",
  19 => "horse",
  20 => "sheep",
  21 => "cow",
  22 => "elephant",
  23 => "bear",
  24 => "zebra",
  25 => "giraffe",
  26 => "hat",
  27 => "backpack",
  28 => "umbrella",
  29 => "shoe",
  30 => "eye glasses",
  31 => "handbag",
  32 => "tie",
  33 => "suitcase",
  34 => "frisbee",
  35 => "skis",
  36 => "snowboard",
  37 => "sports",
  38 => "kite",
  39 => "baseball",
  40 => "baseball",
  41 => "skateboard",
  42 => "surfboard",
  43 => "tennis racket",
  44 => "bottle",
  45 => "plate",
  46 => "wine glass",
  47 => "cup",
  48 => "fork",
  49 => "knife",
  50 => "spoon",
  51 => "bowl",
  52 => "banana",
  53 => "apple",
  54 => "sandwich",
  55 => "orange",
  56 => "broccoli",
  57 => "carrot",
  58 => "hot dog",
  59 => "pizza",
  60 => "donut",
  61 => "cake",
  62 => "chair",
  63 => "couch",
  64 => "potted plant",
  65 => "bed",
  66 => "mirror",
  67 => "dining table",
  68 => "window",
  69 => "desk",
  70 => "toilet",
  71 => "door",
  72 => "tv",
  73 => "laptop",
  74 => "mouse",
  75 => "remote",
  76 => "keyboard",
  77 => "cell phone",
  78 => "microwave",
  79 => "oven",
  80 => "toaster",
  81 => "sink",
  82 => "refrigerator",
  83 => "blender",
  84 => "book",
  85 => "clock",
  86 => "vase",
  87 => "scissors",
  88 => "teddy bear",
  89 => "hair drier",
  90 => "toothbrush",
  91 => "hair brush"
}

def draw_box(img, label, box)
  width, height = img.dimensions

  thickness = 6
  top = (box[0] * height).round - thickness
  left = (box[1] * width).round - thickness
  bottom = (box[2] * height).round + thickness
  right = (box[3] * width).round + thickness

  # draw box
  img.combine_options do |c|
    c.draw "rectangle #{left},#{top} #{right},#{bottom}"
    c.fill "none"
    c.stroke "red"
    c.strokewidth thickness
  end

  # draw text
  img.combine_options do |c|
    c.draw "text #{left},#{top - 5} \"#{label}\""
    c.fill "red"
    c.pointsize 96
  end
end

result["num_detections"].each_with_index do |n, idx|
  n.to_i.times do |i|
    label = result["detection_classes"][idx][i].to_i
    label = coco_labels[label] || label
    box = result["detection_boxes"][idx][i]
    draw_box(img, label, box)
  end
end

img.write("results/#{ARGV[0]}")