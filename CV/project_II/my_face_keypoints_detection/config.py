OPTIMIRZER_ = ["opt_SGD", "opt_Momentum", "opt_RMSprop", "opt_Adam"]

FLAG_RESTORE_MODEL = False
PHASE = 'Predict'  # Predict
MODEL_NAME = r'detector_epoch_400.pt'
MODEL_SAVE_PATH = r'trained_models'
RESULT_IMGS_SAVE_PATH = r'Result_imgs_face_keypoints'
RESULT_TRAIN_LOG_IMGS_SAVE_PATH = r'Result_imgs_train_log'
IMG_TO_PREDICT = r'Img_for_test\1.png'

DEVICE = 'gpu'  # gpu
EPOCH = 10000
BATCH_TRAIN = 64
BATCH_VAL = 64
OPTIMIRZER = OPTIMIRZER_[1]
LEARNING_RATE = 0.001
LOG_INTERVAL = 20
SAVE_MODEL_INTERVAL = 10
NET_IMG_SIZE = (112, 112)