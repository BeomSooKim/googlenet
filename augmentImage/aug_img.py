#%%
import imgaug.augmenters as iaa

aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    #iaa.Sometimes(0.5, iaa.Add((-30, 30))),
    iaa.Sometimes(0.5, iaa.Affine(rotate = (-30, 30),
    shear = (-10, 10),
    scale = (0.9, 1.1),
    translate_percent = {"x":(-0.1, 0.1), "y":(-0.1,0.1)})),
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma = (0.0, 2.0)))
])