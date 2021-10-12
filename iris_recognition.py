from sys import maxsize
import numpy as np
import cv2 as cv2
from numpy.core.arrayprint import array2string
from numpy.lib.shape_base import split
from daugman import daugman
from scipy.spatial import distance
import itertools
import glob
import re
np.set_printoptions(threshold=maxsize)


def daugman_normalizaiton(image, height, width, r_in, r_out):
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / width)  # Theta values
    # Create empty flatten image
    flat = np.zeros((height, width, 3), np.uint8)
    circle_x = int(image.shape[0] / 2)
    circle_y = int(image.shape[1] / 2)

    for i in range(width):
        for j in range(height):
            theta = thetas[i]  # value of theta coordinate
            r_pro = j / height  # value of r coordinate(normalized)

            # get coordinate of boundaries
            Xi = circle_x + r_in * np.cos(theta)
            Yi = circle_y + r_in * np.sin(theta)
            Xo = circle_x + r_out * np.cos(theta)
            Yo = circle_y + r_out * np.sin(theta)

            # the matched cartesian coordinates for the polar coordinates
            Xc = (1 - r_pro) * Xi + r_pro * Xo
            Yc = (1 - r_pro) * Yi + r_pro * Yo

            color = image[int(Xc)][int(Yc)]  # color of the pixel

            flat[j][i] = color
    return flat

def load_baza(files):
    names = []
    files = []
    ids = []
    for r in range(30):
        ar = np.array(r+1)
        cur_path = "UBIRIS_V1_800_600/Sessao_1/" + array2string(ar)
        i = 0
        for file in glob.glob(cur_path + "/*.jpg"):
            if(i > 2):
                i+=1
                continue
            else:  
                files.append(file)
                i+=1
            
        (paths,name,ext) = re.split('\\\\|\.',files[r])
        names.append(name+".jpg")
        (Przed,x,y,z) = name.split('_')
        ids.append(int(x))
    return files, names, ids

def Search(kod,pkod):
    Codes = [code for code in glob.glob("Baza/*.bmp")]
    Hamm = []
    Data = []
    for code in Codes:
        # wczytana baza jest znów zakodowana jako RGB trzeba zamienić na czarno białe
        th, d2 =  cv2.threshold(cv2.imread(code, 0), 0, 255, cv2.THRESH_BINARY)
    
        #porównywanie kodów
        Hamm.append(distance.hamming(d2.ravel(), kod.ravel()))

    #wybór najtrafniejszego kodu (im bliżej zera tym lepiej)
    Min = min(Hamm)
    
    MinM = Codes[Hamm.index(Min)]

    (paths,name,ext) = re.split('\\\\|\.',pkod)
    (Przed,x,y,z) = name.split('_')

    (paths2,name2,ext2) = re.split('\\\\|\.',MinM)
    _,kodID = name2.split('kod')

    SMin = str(np.round(Min,3))
    nameF = name + "." + ext
    nameF2 = name2 + "." + ext2
    _id = int(x)
    _idKOD = int(kodID)
    check = int(_idKOD/3)+1
    if(check == _id):
        sukces = "1"
    else:
        sukces = "0"
    
    print(nameF + "   |    " + nameF2 + "   |    " + SMin + "   |    " + sukces )
    
    Data.append((kod,MinM,Min))

def img_processing(IMG):

    IMG = cv2.medianBlur(IMG, 5)

    # Użycie adatptive threshold dla pozbycia się zbędnych wartości piskeli (binaryzacja)
    th1 = cv2.adaptiveThreshold(IMG, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY, 35, 5)

    # Zamkniecie i otwracie usuwa większąść szumu, lepiej widać zarys oka
    morph = cv2.morphologyEx(th1, cv2.MORPH_OPEN, cv2.getStructuringElement(2, (5, 5)))

    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, cv2.getStructuringElement(2, (3, 3)))

    return morph


files =[]
NotFound = []

# dla bazy
Baza, names, ids = load_baza(files)

# dla porównania zdjęć z bazą
Files = [file for file in glob.glob("foty/*.jpg")]

# Tutaj należy przypisać zmienną Baza lub Files w zależności czy chcemy kodować bazę czy wykrywać 
files = Files


#----------------------------------------
#-----------GŁÓWNA PĘTLA-----------------
#----------------------------------------

print("Badane Zdjęcie"+"   |    "+"Znaleziony KOD"+"   |    "+"Podobieństwo"+"   |    "+"Sukces?")
for file in files:

    # wczytanie zdjęcia do zmiennej
    img = cv2.imread(file, 0)
    cimg = cv2.imread(file)

    # przetworzenie zdjęcia
    processed = img_processing(img.copy())

    # Wbudowana funkcja Hougha do znajdowania okręgów
    circles = cv2.HoughCircles(processed, cv2.HOUGH_GRADIENT_ALT, 2.5, 50,
                            param1=100, param2=0.6, minRadius=25, maxRadius=60)
   
    if circles is not None:
        radius = 0
        r = 0
        filtered_img = []

        for i in circles[0, :]:

            # narysuj znalezione źrenice
            cv2.circle(cimg, (i[0].astype(int), i[1].astype(int)), i[2].astype(int), (0, 0, 0), 3)

            # Jeżeli znalazł źrenicę wykorzystuje Daugmana żeby znaleść zarys tęczówki. 
            # Działa dobrze jeżeli znamy prawdopodobny środek i promień. (Jest mocno obliczenio żerny)

            Cx = i[0].astype(int)
            Cy = i[1].astype(int)
            start_r = 90

            # będziemy szukać w kwadracie o środku Cx, Cy i pewnych wymiarach (tutaj 9x9)
            # wyszukuje co 2 pixel
            a = range(Cx - 3, Cx + 3, 2)
            b = range(Cy - 3, Cy + 3, 2)
            all_points = itertools.product(a, b)

            values = []
            coords = []
            
            for p in all_points:
                tmp = daugman(p, start_r, img)
                if tmp is not None:
                    val, circle = tmp
                    values.append(val)
                    coords.append(circle)

            # zwraca wartość okręgu o największej intensywności
            center, radius = coords[values.index(max(values))]

            cv2.circle(cimg, center, radius, (0, 0, 0), 3)

            r = i[2].astype(int)

        # tworzenie wstęgi ze znalezionej tęczówki
        image_nor = daugman_normalizaiton(img.copy(), 90, 360, r, radius)

        # kilkuktrona filtracja po kącie
        As = [0,30,60,90,120,150]

        filtered_img = np.zeros(image_nor.shape, np.float32)
        for A in As:
            #cv.getGaborKernel(ksize, sigma, theta, lambd, gamma[, psi[, ktype]])
            g_kernel = cv2.getGaborKernel((27, 27), 3.4, np.pi*A/180, 8.1, 0.9, 0, ktype=cv2.CV_32F)
            _filtered_img = cv2.filter2D(image_nor, cv2.CV_8UC3, g_kernel)
            filtered_img += _filtered_img
        
        # normalizacja filtru
        filtered_img = filtered_img / filtered_img.max() *255
        filtered_img = filtered_img.astype(np.uint8)

        # uzyskanie z filtracji kodu dwuwymiarowego i zbinaryzowanego 
        filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
        filtered_img = cv2.adaptiveThreshold(filtered_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY, 19, 1)  

        # obsługa zapisu kodów lub wyszukiwania
        if Baza.count(file)==True:
            
            cv2.imwrite("Baza/kod"+array2string(np.array(files.index(file)))+".bmp",filtered_img)
            print(file)
        else:
            Search(filtered_img, file)

    else:
        NotFound.append(file)
        #print(file + " ----> błąd:  nie wykryto tęczówki")
        continue

cv2.waitKey(0)
cv2.destroyAllWindows()