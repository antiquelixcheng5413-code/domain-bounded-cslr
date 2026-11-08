import glob
import cv2

p = sorted(glob.glob("/home/su127/FYP/domain-bounded-cslr/artifacts/official_rgb/dev/*/*/*.jpg"))
print("n_jpg", len(p))
for f in p[:2] + p[-2:]:
    im = cv2.imread(f)
    print(f.split("/dev/")[-1], None if im is None else im.shape)
