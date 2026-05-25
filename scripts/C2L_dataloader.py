import random
import pandas as pd
import torch
import torch.utils.data as data
from scipy import ndimage
import numpy as np
import cv2


class Cine2LGEDataset(data.Dataset):
    def __init__(self, csv_data_path, root, is_trans, img_size):
        self.img_size = img_size
        self.is_trans = is_trans
        self.root = root
        self.data = pd.read_csv(csv_data_path, encoding='utf-8')

        self.folder = self.data['folder'].values.tolist()
        self.patient = self.data['patient'].values.tolist()
        self.npz_name = self.data['npz_name'].values.tolist()
        self.Pcine_index = self.data['Pcine_index'].values.tolist()
        self.Mf = self.data['Mf'].values.tolist()
        self.Mfs = self.data['Mfs'].values.tolist()
        self.size = len(self.data)
        self.clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))

    def __len__(self):
        return self.size

    def normalization(self, img):
        img_normal = (img - img.min()) / (img.max() - img.min())
        return img_normal  # Image.fromarray(np.uint8(img_normal * 255))

    def adjust_gamma(self, image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def get_fft_image(self, img_gray):
        f = np.fft.fft2(img_gray)

        fshift = np.fft.fftshift(f)
        rows, cols = fshift.shape
        mid_x, mid_y = int((rows) / 2), (int((cols) / 2))

        mask1 = np.ones((rows, cols), dtype=np.uint8)
        mask1[mid_x - 1:mid_x + 1, mid_y - 1:mid_y + 1] = 0
        fshift1 = mask1 * fshift
        isshift1 = np.fft.ifftshift(fshift1)

        mask2 = np.zeros((rows, cols), dtype=np.uint8)
        mask2[mid_x - 30:mid_x + 30, mid_y - 30:mid_y + 30] = 1
        fshift2 = mask2 * fshift
        isshift2 = np.fft.ifftshift(fshift2)

        high = np.fft.ifft2(isshift1)
        low = np.fft.ifft2(isshift2)

        img_high = np.abs(high)
        img_low = np.abs(low)
        return img_high, img_low

    def get_transformer_enhance(self, data, angle):
        def rotate_image(yuan_img, angle=90):
            rotated_array = ndimage.rotate(yuan_img, angle=angle, reshape=False)
            return rotated_array

        def rotate_image_multi(crop_cine, angle=90):
            all_cine = []
            x, y, z = crop_cine.shape
            min_idx = min(x, y, z)
            if min_idx == z:
                crop_cine = np.transpose(crop_cine, (2, 0, 1))
            for i in range(min_idx):
                nor = rotate_image(crop_cine[i], angle)
                all_cine.append(nor)
            crop_cine = np.stack(all_cine, axis=0)
            if min_idx == z:
                crop_cine = np.transpose(crop_cine, (1, 2, 0))
            return crop_cine

        keys = data.files
        # angle = 180  # (-90,0,90,180)
        all_image = []
        for i in range(len(keys)):
            temp_image = data[keys[i]]
            new_image = None
            if len(temp_image.shape) == 2:
                new_image = rotate_image(temp_image, angle=angle)
            if len(temp_image.shape) == 3:
                new_image = rotate_image_multi(temp_image, angle=angle)
            all_image.append(new_image)
        ro_data = {}
        for j in range(len(keys)):
            ro_data[keys[j]] = all_image[j]
        return ro_data

    def get_style_label(self, mf, mfs):
        label = 0
        if mf == 'PHI':
            if mfs == float(1.5):
                label = 0
            else:
                label = 3
        elif mf == 'SIE':
            label = 2
        elif mf == 'GE':
            label = 1
        elif mf == 'UIH':
            label = 3
        return label

    def normalize_to_255(self, img, min_raw, max_raw):
        return ((img - min_raw) / (max_raw - min_raw) * 255).clip(0, 255).astype(np.uint8)

    def apply_window(self, img_norm, ww=200, wl=128):
        lower = wl - ww / 2
        upper = wl + ww / 2
        return np.clip((img_norm - lower) / ww * 255, 0, 255).astype(np.uint8)

    def __getitem__(self, index):
        data = np.load(self.root + '/' + self.folder[index] + '/' + self.npz_name[index])
        if self.is_trans:
            angle = random.choice([-90, 0, 90, 180, 0, 0, 0])
            data = self.get_transformer_enhance(data, angle)
        cine_image = data['cine']
        lge_image = data['lge']
        t2_image = data['t2']
        lge_heart_mask = data['lge_heart_mask']  # 255
        map_myo_motion = data['map_myo_motion']  # 012
        map_myo_color = data['map_myo_color']  # 012
        cine_minus = data['cine_minus']
        cine_plus = data['cine_plus']

        cine_myo = data['cine_myo']  # 1   (128, 128, 25)
        cine_minus_myo = data['cine_minus_myo']
        cine_plus_myo = data['cine_plus_myo']

        cine_only_myo = np.transpose(cine_myo, (2, 0, 1)) * cine_image  # (25,128,128)
        cine_myo_minus = np.transpose(cine_minus_myo, (2, 0, 1)) * cine_minus
        cine_myo_plus = np.transpose(cine_plus_myo, (2, 0, 1)) * cine_plus

        label = self.get_style_label(self.Mf[index], self.Mfs[index])
        cine_right = cine_image[self.Pcine_index[index]]

        # 归一化
        nor_lge_image = self.normalization(lge_image)
        nor_t2_image = self.normalization(t2_image)
        nor_cine_right = self.normalization(cine_right)

        nor_cine_only_myo = self.normalization(cine_only_myo)
        nor_cine_only_myo_minus = self.normalization(cine_myo_minus)
        nor_cine_only_myo_plus = self.normalization(cine_myo_plus)

        # fft
        img_high, img_low = self.get_fft_image(nor_t2_image)
        cl_image = self.clahe.apply(np.uint8(nor_t2_image * 255))
        gamma_corrected_t2 = self.adjust_gamma(cl_image, gamma=0.6) / 255  # 0~1
        # t2 fft
        img_high = self.normalization(img_high) * 2 - 1

        # 直接使用tensor
        nor_cine_right = nor_cine_right * 2 - 1
        nor_lge_image = nor_lge_image * 2 - 1
        nor_t2_image = nor_t2_image * 2 - 1
        gamma_corrected_t2 = gamma_corrected_t2 * 2 - 1
        img_high = img_high * 2 - 1

        cine_right = torch.tensor(nor_cine_right.copy()).unsqueeze(0)
        cine_LH = torch.tensor(nor_cine_only_myo.copy()).unsqueeze(0)
        cine_LH_minus = torch.tensor(nor_cine_only_myo_minus.copy()).unsqueeze(0)
        cine_LH_plus = torch.tensor(nor_cine_only_myo_plus.copy()).unsqueeze(0)

        lge = torch.tensor(nor_lge_image.copy()).unsqueeze(0)  # [1,128,128]
        t2 = torch.tensor(nor_t2_image.copy()).unsqueeze(0)
        t2_high = torch.tensor(img_high.copy()).unsqueeze(0)
        gamma_t2 = torch.tensor(gamma_corrected_t2.copy()).unsqueeze(0)  # [1,128,128]

        heart_label = torch.tensor(lge_heart_mask // 255)  # [1,128,128]

        map_myo_motion = torch.tensor(map_myo_motion - 1).unsqueeze(0)
        nor_map_myo_color = self.normalization(map_myo_color) * 2 - 1
        map_myo_color = torch.tensor(nor_map_myo_color).unsqueeze(0)

        return {'cine_right': cine_right.float(), 'lge': lge.float(), 't2': t2.float(),
                'lge_heart_label': heart_label.float(),
                'map_myo_motion': map_myo_motion.float(), 'map_myo_color': map_myo_color.float(),
                't2_gamma': gamma_t2.float(), 't2_fft': t2_high.float(),
                'cine_LH': cine_LH.float(), 'cine_LH_minus': cine_LH_minus.float(),
                'cine_LH_plus': cine_LH_plus.float(),
                'npz_name': self.npz_name[index], 'label': torch.tensor(label)}


def divide_data(all_data_path, save_train, save_val, save_test):
    dataframe = pd.read_csv(all_data_path, encoding="utf-8")
    image_info = dataframe["patient"].values.tolist()
    image_info = list(set(image_info))
    random.seed(42)
    random.shuffle(image_info)
    scale = [8, 0, 2]
    fen = 10
    train_ID = image_info[:int(len(image_info) * (scale[0] / fen))]
    val_ID = image_info[int(len(image_info) * (scale[0] / fen)):int(len(image_info) * ((scale[0] + scale[1]) / fen))]
    test_ID = image_info[int(len(image_info) * ((scale[0] + scale[1]) / fen)):]

    train_info = dataframe[dataframe["patient"].isin(train_ID)]
    val_info = dataframe[dataframe["patient"].isin(val_ID)]
    test_info = dataframe[dataframe["patient"].isin(test_ID)]

    train_info.to_csv(save_train, index=None)
    val_info.to_csv(save_val, index=None)
    test_info.to_csv(save_test, index=None)


if __name__ == '__main__':
    divide_data(all_data_path="E:/2025Project/AMI_ddpm/data_excel/data_PLA_npz.csv",
                save_train="E:/2025Project/AMI_ddpm/data_excel/data_PLA_npz_train.csv",
                save_val="E:/2025Project/AMI_ddpm/data_excel/data_PLA_npz_val.csv",
                save_test="E:/2025Project/AMI_ddpm/data_excel/data_PLA_npz_test.csv")
