import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import model

DIR="data/models/"
MODEL="model-30-epochs-l1-adam-003-lr-kcpu.pth"

def get_model(PATH, model):
    device = torch.device('cpu')
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.eval()
    return model

def load_apply_preprocessing(PATH):

    img_gray = cv2.imread(PATH)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (150, 150))
    img_gray = img_gray.astype('float32') / 255.0
    img_gray = torch.Tensor(img_gray)
    img_gray = torch.unsqueeze(img_gray, 0)
    img_gray = torch.unsqueeze(img_gray, 0)

    return img_gray

def show_image(img_path, img_gray, pred_img):
    
    img_gray = torch.squeeze(torch.squeeze(img_gray, 0), 0).numpy()
    img_gray = np.transpose(np.array([img_gray, img_gray, img_gray]), (1, 2, 0))
    pred_img = torch.squeeze(pred_img, 0)
    pred_img = np.transpose(pred_img, (1, 2, 0))
    img_stack = np.hstack([img_gray, pred_img])
    plt.figure(figsize=(10, 5), dpi=100)
    plt.title(img_path)
    plt.imshow(img_stack)
    plt.show()

def predict(model, img):
    with torch.no_grad():
        pred = model(img)
    return pred

if __name__ == "__main__":

    img_path = sys.argv[1] #"data/sample-images/forest.jpg"
    model = model.AutoEncoder()
    model = get_model(DIR+MODEL, model)
    
    tnsr_img = load_apply_preprocessing(img_path)
    pred_img = predict(model, tnsr_img)
    show_image(img_path, tnsr_img, pred_img)