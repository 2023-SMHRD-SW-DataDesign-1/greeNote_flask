# 모듈 구성 및 임포트
import tensorflow as tf
import IPython.display as display
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False
from fixed import tensor_to_image, load_img, extractor, style_extractor, train_step
import base64

# 원본이미지, 스타일이미지 불러오기
content_image = load_img('C:\\Users\\smhrd\\Desktop\\HGY\\Flask\\chanoAI\\원본\\plant_04.jpg')
style_image = load_img('C:\\Users\\smhrd\\Desktop\\HGY\\Flask\\chanoAI\\명화\\피카소.jpg')

# VGG19 모델을 불러오기
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')



style_outputs = style_extractor(style_image*255)

results = extractor(tf.constant(content_image))

# 스타일과 콘텐츠의 타깃값을 지정
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

# 최적화시킬 이미지를 담을 tf.Variable을 정의하고 콘텐츠 이미지로 초기화
# (이때 tf.Variable는 콘텐츠 이미지와 크기가 같아야 합니다.)
image = tf.Variable(content_image)

# 최적화를 진행
import time
start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image, style_targets, content_targets)
    print(".", end='', flush=True)
  display.clear_output(wait=True)
  display.display(tensor_to_image(image))
  print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end-start))
image_data = image.numpy()
image_base64 = base64.b64encode(image_data).decode('utf-8')
