import seed
from utils import *
from models import *
from preprocessing import Preprocessing


X, y, files = load_data(patches=True)
X = X - 0.5

pre = Preprocessing(standardize=False, samplewise=False)
X_train, X_valid, y_train, y_valid = pre.split_data(
    X, y, test_size=0.1, shuffle=True)


conf = Config(epochs=1000, patience=50,
              use_class_weights=True, batch_size=2000)
basic_fcn = BasicFCN(config=conf)
basic_fcn.train(X_train, y_train, X_valid, y_valid)


# Prediction
X_test, test_files = load_tests()
# X_test2 = pre.transform(X_test)
X_test = X_test - 0.5
X_pred = basic_fcn.predict(X_test, test_files)
vis_pred(X_test, X_pred, last_n=False, img_sz=608)

# submission
# dir_path = "runs/BasicFCN_20190626-095011/predictions"
dir_path = os.path.join(basic_fcn.model_dir, basic_fcn.config.pred_dir)
pattern = os.path.join(dir_path, "test_*.png")
submission_filename = os.path.join(dir_path, "submission.csv")
image_filenames = glob.glob(pattern)
for image_filename in image_filenames:
    print(image_filename)
masks_to_submission(submission_filename, *image_filenames)
