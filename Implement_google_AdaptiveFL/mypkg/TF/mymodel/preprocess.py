import tensorflow as tf
class GoogleAdaptive_tfds_preprocess():
    def __init__(self):
        self.dataset=None
        self.rng=None
        self.train=None
        self.global_seed=None
        self.crop_size=None # 欲裁切的大小
        self.batch_zize=None
        self.shuffle_buffer=None
        self.prefetch_buffer=None

    def preprocess(self, dataset, rng, train=True,
                   global_seed=2021, crop_size=24, 
                   batch_zize=20,
                   shuffle_buffer=100, prefetch_buffer=20,
                  ):
        """
        輸入
        一個client的資料集，內部進行shuffle、batching、preprocessing
        輸出
        一個標記如何切割、epoch、batching、shuffle、預處理後的 
        single client Dataset:tf.PrefetchDataset:tuple(batched)"""
        self.dataset=dataset
        self.rng=rng
        self.train=train
        self.global_seed=global_seed
        self.crop_size=crop_size # 欲裁切的大小
        self.batch_zize=batch_zize
        self.shuffle_buffer=shuffle_buffer
        self.prefetch_buffer=prefetch_buffer
        
        def simlpe_rescale(image, label):
            '''基本圖像正規化 (batch_format_fn)'''
            # for IMAGE
            image = tf.cast(image, tf.float32)
            # 論文要求的 Z-score 正規化 (mean and standard deviation)
            image_mean = tf.math.reduce_mean(image)
            image_std = tf.math.reduce_std(image,1)
            image = ((image-image_mean) / image_std)

            # for LABEL
            tf.reshape(label, [-1, 1])
            return image, label

        def custom_augment(element):
            '''自訂資料增強 (batch_format_fn)'''
            (image, label) = (element['image'], element['label'])

            # 套用基本圖像正規化
            image, label = simlpe_rescale(image, label)
            
            # 像素正規化
            # 只有 Training 才套用
            if(train):               
                # 隨機裁切成 crop_size 隨機水平翻轉
                seed = rng.make_seeds(2)[0]
                image = tf.image.stateless_random_crop(image, size=[self.crop_size, self.crop_size, 3], seed=seed)
                image = tf.image.stateless_random_flip_left_right(image, seed=seed)

            # Testing時套用
            else:
                # Centrally Cropping (此函式的裁切自動裁中心)
                image = tf.image.resize_with_crop_or_pad(image, self.crop_size, self.crop_size)
            return (image, label) 

        return dataset.shuffle(shuffle_buffer, seed=global_seed).map(
            custom_augment, num_parallel_calls=tf.data.AUTOTUNE).batch(
            batch_zize).prefetch(prefetch_buffer)